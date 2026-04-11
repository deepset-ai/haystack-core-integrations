"""End-to-end test with a real HuggingFace dataset and real embeddings.

Uses:
  - Dataset: `squad` (Stanford QA) — real Wikipedia passages
  - Embedder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast, no API key)
  - Store: OracleDocumentStore → Oracle AI Database 26ai on OCI Free Tier

Run:
    ORACLE_USER=ADMIN \
    ORACLE_PASSWORD=... \
    ORACLE_DSN=deepresearch_low \
    ORACLE_WALLET_LOCATION=~/.oracle/wallet_deepresearch \
    ORACLE_WALLET_PASSWORD=... \
    python tests/e2e_real_data.py
"""

import os
import time

from datasets import load_dataset
from fastembed import TextEmbedding
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever
from haystack_integrations.document_stores.oracle import (
    OracleConnectionConfig,
    OracleDocumentStore,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TABLE = "e2e_squad_test"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, ONNX, ~25MB, no PyTorch
EMBED_DIM = 384
N_DOCS = 200  # subset of SQuAD to keep it fast
TOP_K = 5

QUERIES = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "How does photosynthesis work?",
    "What caused the First World War?",
    "Who wrote Romeo and Juliet?",
]


def build_store() -> OracleDocumentStore:
    return OracleDocumentStore(
        connection_config=OracleConnectionConfig(
            user=os.environ["ORACLE_USER"],
            password=Secret.from_env_var("ORACLE_PASSWORD"),
            dsn=os.environ["ORACLE_DSN"],
            wallet_location=os.environ.get("ORACLE_WALLET_LOCATION"),
            wallet_password=(
                Secret.from_env_var("ORACLE_WALLET_PASSWORD") if os.environ.get("ORACLE_WALLET_PASSWORD") else None
            ),
        ),
        table_name=TABLE,
        embedding_dim=EMBED_DIM,
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )


def load_squad_passages(n: int) -> list[dict]:
    """Load unique Wikipedia passages from SQuAD validation set."""
    print(f"Loading SQuAD dataset (first {n} unique passages)...")
    ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)
    seen, passages = set(), []
    for row in ds:
        ctx = row["context"].strip()
        if ctx not in seen:
            seen.add(ctx)
            passages.append(
                {
                    "text": ctx,
                    "title": row["title"],
                    "id": row["id"],
                }
            )
        if len(passages) >= n:
            break
    print(f"  Loaded {len(passages)} unique passages")
    return passages


def embed(model: TextEmbedding, texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in model.embed(texts)]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = TextEmbedding(model_name=EMBED_MODEL)
    print("  Model loaded")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    passages = load_squad_passages(N_DOCS)

    # ------------------------------------------------------------------
    # 3. Embed
    # ------------------------------------------------------------------
    print(f"\nEmbedding {len(passages)} passages...")
    t0 = time.perf_counter()
    texts = [p["text"] for p in passages]
    embeddings = embed(model, texts)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Build Haystack Documents
    # ------------------------------------------------------------------
    documents = [
        Document(
            content=p["text"],
            meta={"title": p["title"], "squad_id": p["id"]},
            embedding=emb,
        )
        for p, emb in zip(passages, embeddings, strict=True)
    ]

    # ------------------------------------------------------------------
    # 5. Connect to Oracle and write
    # ------------------------------------------------------------------
    print(f"\nConnecting to Oracle ADB ({os.environ['ORACLE_DSN']})...")
    store = build_store()
    print(f"  Connected — table: {TABLE}")

    print(f"\nWriting {len(documents)} documents...")
    t0 = time.perf_counter()
    written = store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
    elapsed = time.perf_counter() - t0
    print(f"  Written: {written} docs in {elapsed:.1f}s ({written / elapsed:.0f} docs/sec)")

    total = store.count_documents()
    print(f"  Total in table: {total}")

    # ------------------------------------------------------------------
    # 6. Create HNSW index
    # ------------------------------------------------------------------
    print("\nCreating HNSW index...")
    t0 = time.perf_counter()
    store.create_hnsw_index()
    print(f"  Index created in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 7. Query
    # ------------------------------------------------------------------
    retriever = OracleEmbeddingRetriever(document_store=store, top_k=TOP_K)

    print(f"\n{'=' * 60}")
    print("RETRIEVAL RESULTS")
    print(f"{'=' * 60}")

    for query in QUERIES:
        query_emb = list(model.embed([query]))[0].tolist()
        t0 = time.perf_counter()
        result = retriever.run(query_embedding=query_emb)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f'\nQuery: "{query}"  [{latency_ms:.0f}ms]')
        print("-" * 60)
        for i, doc in enumerate(result["documents"], 1):
            snippet = doc.content[:120].replace("\n", " ")
            print(f"  {i}. [{doc.score:.4f}] {doc.meta['title']}")
            print(f"     {snippet}...")

    # ------------------------------------------------------------------
    # 8. Filter test — only return passages about a specific topic
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("FILTERED RETRIEVAL — only 'Beyoncé' passages")
    print(f"{'=' * 60}")
    query = "Who is Beyoncé?"
    query_emb = list(model.embed([query]))[0].tolist()
    result = retriever.run(
        query_embedding=query_emb,
        filters={"field": "meta.title", "operator": "==", "value": "Beyoncé"},
    )
    if result["documents"]:
        for doc in result["documents"]:
            print(f"  [{doc.score:.4f}] {doc.content[:150].replace(chr(10), ' ')}...")
    else:
        print("  (no Beyoncé passages in this subset)")

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    answer = input("Drop test table? [y/N] ").strip().lower()
    if answer == "y":
        with store._get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP TABLE {TABLE} PURGE")
            conn.commit()
        print(f"  Table {TABLE} dropped.")
    else:
        print(f"  Table {TABLE} kept — {total} docs remain in Oracle.")


if __name__ == "__main__":
    main()
