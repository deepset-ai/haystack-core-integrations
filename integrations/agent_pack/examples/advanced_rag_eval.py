# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Mini evaluation harness for the advanced RAG agent.

Runs the agent on a set of questions with known ground truth and reports, per case:

- **Retrieval correctness**: the documents the agent actually saw (the run's accumulated
  `documents` output) are checked against a ground-truth predicate over document metadata. On
  the small corpus the expected set is tiny and enumerable, so we check **recall**
  (did it find all expected docs?). On the large corpus the expected set is huge, so we check
  **constraint precision** (does every retrieved doc satisfy the constraints the question
  implies? — this encodes the expectation that the agent expresses constraints as filters
  instead of hoping the query text does the narrowing).
- **Process metrics**: whether `list_metadata_fields` was called before any retrieval (the core
  behaviour validated for deepset-ai/haystack#11000), per-tool call counts, how many retrieval calls used a
  filter, error tool results (e.g. malformed filters, over-broad fetches), agent steps, and
  wall-clock time.
- **Efficiency budget**: per-case upper bounds on metadata-inspection and retrieval calls — a
  case fails when the agent re-inspects or flail-retries beyond the budget, even if the answer
  is right. The budgets are deliberately lenient: erring on the side of too many retrievals is
  preferred over too few, so only genuine flailing should trip them.
- **Adversarial cases**: questions about fields/values that don't exist in the corpus; the
  correct behaviour is to discover the absence via the metadata tools and say so, not to
  hallucinate an answer (checked with a crude negation heuristic — an LLM judge would do this
  more robustly).
- **Answer sanity**: optional keywords the final answer must mention.
- **Citation resolution**: every `[doc <short-id>]` reference in the answer must resolve to a
  document present in the run's returned `documents` list (no hallucinated references).
- **Token usage**: per case and totalled, for comparing models/reasoning efforts.

Run from the integration directory (`integrations/agent_pack`) with `OPENAI_API_KEY` set:

    hatch run test:python examples/advanced_rag_eval.py small
    hatch run test:python examples/advanced_rag_eval.py large   # needs `datasets`, downloads reviews from HF

To validate against a real database instead of `InMemoryDocumentStore`, pass `--store opensearch`
(needs `opensearch-haystack`). Set `OPENSEARCH_URL` if not http://localhost:9200, and
`OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` for a security-enabled instance (use an https:// URL
then). Start one locally with e.g.
`docker run -p 9200:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch:2`:

    hatch run test:python examples/advanced_rag_eval.py small --store opensearch
    hatch run test:python examples/advanced_rag_eval.py large --store opensearch

The extra dependencies (`datasets`, `opensearch-haystack`) can be installed into the hatch test
environment with `uv pip install datasets opensearch-haystack` from inside `hatch -e test shell`.
An already-populated OpenSearch index is reused, so repeat runs skip re-indexing.
"""

import argparse
import itertools
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent

RETRIEVAL_TOOLS = ("search_documents", "fetch_documents_by_filter")
METADATA_TOOLS = ("list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range")
_CITATION_RE = re.compile(r"\[doc ([0-9a-f]{4,16})\]")
# The system prompt instructs the agent to begin with this phrase when nothing matches; the regex
# is a crude fallback for non-compliant answers (an LLM judge would do this properly).
_ABSENCE_PHRASE = "no matching information was found"
_ABSENCE_RE = re.compile(
    r"\b(no|not|none|nothing|cannot|can't|couldn't|don't|doesn't|isn't|aren't|unable|unfortunately|missing|absent)\b"
)


def _acknowledges_absence(answer: str) -> bool:
    """
    Whether the answer acknowledges that the requested information is absent from the corpus.

    :param answer: The agent's final answer.
    :returns: True if the canonical absence phrase (or, as a fallback, a negation word) is present.
    """
    lowered = answer.lower()
    return _ABSENCE_PHRASE in lowered or bool(_ABSENCE_RE.search(lowered))


@dataclass
class EvalCase:
    """One evaluation case: a question plus ground truth."""

    question: str
    # Ground truth: which documents answer the question, as a predicate over `Document.meta`.
    predicate: Callable[[dict[str, Any]], bool]
    # Check recall of the full expected set (small corpora only — the expected set must be enumerable).
    check_recall: bool = True
    # Minimum number of ground-truth-matching documents the agent must have retrieved.
    min_docs: int = 1
    # Keywords (case-insensitive) the final answer must mention.
    answer_must_mention: tuple[str, ...] = ()
    # Minimum constraint precision, gated only when `check_recall` is False (large corpora).
    # Below 1.0 by design: extra broader retrievals are acceptable as long as the constrained
    # documents dominate what the agent saw.
    min_precision: float = 0.5
    # Adversarial: the question asks about fields/values absent from the corpus; pass = the answer
    # acknowledges the absence (retrieval metrics are skipped, the predicate matches nothing).
    expect_absent: bool = False
    # Efficiency budgets: maximum metadata-inspection and retrieval tool calls per run. Lenient on
    # purpose — too many retrievals is better than too few.
    max_metadata_calls: int = 5
    max_retrieval_calls: int = 5


@dataclass
class RunStats:
    """Tool-level statistics extracted from one agent run."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errors: int = 0

    @property
    def inspected_first(self) -> bool:
        """Whether `list_metadata_fields` was called before any retrieval tool."""
        for name, _ in self.calls:
            if name == "list_metadata_fields":
                return True
            if name in RETRIEVAL_TOOLS:
                return False
        return False

    @property
    def filtered_retrieval_calls(self) -> int:
        """Number of retrieval tool calls that included a metadata filter."""
        return sum(1 for name, args in self.calls if name in RETRIEVAL_TOOLS and args.get("filters"))

    @property
    def metadata_calls(self) -> int:
        """Number of metadata-inspection tool calls."""
        return sum(1 for name, _ in self.calls if name in METADATA_TOOLS)

    @property
    def retrieval_calls(self) -> int:
        """Number of retrieval tool calls."""
        return sum(1 for name, _ in self.calls if name in RETRIEVAL_TOOLS)


SMALL_CASES = [
    EvalCase(
        question="What scientific breakthroughs happened after 2015, according to the documents?",
        predicate=lambda m: m.get("category") == "science" and m.get("year", 0) > 2015,
        answer_must_mention=("quantum", "CRISPR"),
    ),
    EvalCase(
        question="Which historical events in the documents happened before 1990?",
        predicate=lambda m: m.get("category") == "history" and m.get("year", 9999) < 1990,
        answer_must_mention=("Berlin", "Apollo"),
    ),
    EvalCase(
        question="What does the German-language document describe?",
        predicate=lambda m: m.get("language") == "de",
        answer_must_mention=("Champions League",),
    ),
    EvalCase(
        question="What does the single highest-rated document describe?",
        predicate=lambda m: m.get("rating") == 5.0,
        answer_must_mention=("Apollo",),
    ),
    EvalCase(
        question="What do the documents say about sports events from 2020 onwards?",
        predicate=lambda m: m.get("category") == "sports" and m.get("year", 0) >= 2020,
        answer_must_mention=("Argentina",),
    ),
    EvalCase(
        question="What is CRISPR used for according to the documents?",
        predicate=lambda m: m.get("category") == "science" and m.get("year") == 2021,
        answer_must_mention=("blindness",),
    ),
    # Adversarial: nonexistent category / language value — the agent should discover the absence.
    EvalCase(
        question="What do the documents in the 'food' category say about cooking?",
        predicate=lambda _m: False,
        expect_absent=True,
    ),
    EvalCase(question="Which of the documents are written in French?", predicate=lambda _m: False, expect_absent=True),
]

LARGE_CASES = [
    EvalCase(
        question=(
            "What do verified purchasers complain about in low-rated (1-2 star) "
            "beauty product reviews from 2022 or later?"
        ),
        predicate=lambda m: (
            m.get("category") == "All_Beauty"
            and m.get("rating", 5.0) <= 2.0
            and m.get("year", 0) >= 2022
            and m.get("verified_purchase") is True
        ),
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="What did reviewers think of digital music purchases before 2010?",
        predicate=lambda m: m.get("category") == "Digital_Music" and m.get("year", 9999) < 2010,
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="Summarize what the most helpful health product reviews (10 or more helpful votes) say.",
        predicate=lambda m: m.get("category") == "Health_and_Personal_Care" and m.get("helpful_vote", 0) >= 10,
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="What are common themes in 5-star beauty product reviews from 2020?",
        predicate=lambda m: m.get("category") == "All_Beauty" and m.get("rating") == 5.0 and m.get("year") == 2020,
        check_recall=False,
        min_docs=3,
    ),
    # Adversarial: nonexistent category / future year — the agent should discover the absence.
    EvalCase(
        question="What do reviews in the Electronics category say about laptop battery life?",
        predicate=lambda _m: False,
        check_recall=False,
        expect_absent=True,
    ),
    EvalCase(
        question="What do beauty product reviews from 2030 or later say?",
        predicate=lambda _m: False,
        check_recall=False,
        expect_absent=True,
    ),
]


def extract_run_stats(messages: list[ChatMessage]) -> RunStats:
    """
    Extract tool calls and error results from an agent run.

    :param messages: The messages returned by `agent.run(...)`.
    :returns: The extracted statistics.
    """
    stats = RunStats()
    for message in messages:
        stats.calls.extend((tc.tool_name, tc.arguments or {}) for tc in message.tool_calls)
        stats.errors += sum(1 for res in message.tool_call_results if res.error)
    return stats


def _sum_usage(total: dict[str, int], usage: dict[str, Any]) -> dict[str, int]:
    """
    Accumulate the numeric entries of a token-usage dict into a running total.

    :param total: The running totals, updated in place.
    :param usage: One run's `token_usage` dict (key names vary by generator).
    :returns: The updated totals.
    """
    for key, value in (usage or {}).items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
    return total


def evaluate_case(agent: Agent, documents_by_id: dict[str, Any], case: EvalCase) -> dict[str, Any]:
    """
    Run the agent on one case and print its report.

    :param agent: The agent under evaluation.
    :param documents_by_id: All documents in the store, keyed by id (the ground-truth universe).
    :param case: The case to evaluate.
    :returns: A dict with `passed` (bool), `usage` (the run's token_usage dict), and `time` (s).
    """
    started = time.perf_counter()
    result = agent.run(messages=[ChatMessage.from_user(case.question)])
    elapsed = time.perf_counter() - started

    stats = extract_run_stats(result["messages"])
    answer = result["last_message"].text or ""
    usage = result.get("token_usage") or {}

    expected_ids = {doc_id for doc_id, doc in documents_by_id.items() if case.predicate(doc.meta)}
    retrieved_docs = result.get("documents") or []
    retrieved_ids = {d.id for d in retrieved_docs}
    matching = [d for d in retrieved_docs if case.predicate(d.meta)]

    recall = len(expected_ids & retrieved_ids) / len(expected_ids) if expected_ids else 0.0
    precision = len(matching) / len(retrieved_docs) if retrieved_docs else 0.0
    mentions_ok = all(kw.lower() in answer.lower() for kw in case.answer_must_mention)

    # Every [doc <short-id>] reference in the answer must resolve to a returned document.
    cited_refs = _CITATION_RE.findall(answer)
    resolved = [ref for ref in cited_refs if any(d.id.startswith(ref) for d in retrieved_docs)]
    citations_ok = len(resolved) == len(cited_refs)

    within_budget = (
        stats.metadata_calls <= case.max_metadata_calls and stats.retrieval_calls <= case.max_retrieval_calls
    )
    if case.expect_absent:
        correct = _acknowledges_absence(answer)
    else:
        correct = (
            len(matching) >= case.min_docs
            and (recall == 1.0 if case.check_recall else precision >= case.min_precision)
            and mentions_ok
            and citations_ok
        )
    passed = stats.inspected_first and within_budget and correct

    counts = Counter(name for name, _ in stats.calls)
    filters_used = [args["filters"] for name, args in stats.calls if name in RETRIEVAL_TOOLS and args.get("filters")]
    print(f"\n[{'PASS' if passed else 'FAIL'}] {case.question}")
    print(f"  tools: {dict(counts)}")
    print(
        f"  inspected-first={stats.inspected_first}  filtered-retrievals={stats.filtered_retrieval_calls}"
        f"  errors={stats.errors}  steps={result['step_count']}  time={elapsed:.1f}s"
    )
    print(
        f"  budget: metadata {stats.metadata_calls}/{case.max_metadata_calls}, "
        f"retrieval {stats.retrieval_calls}/{case.max_retrieval_calls} -> {'ok' if within_budget else 'EXCEEDED'}"
    )
    if case.expect_absent:
        print(f"  expect-absent: acknowledged={correct}")
    else:
        print(
            f"  retrieved={len(retrieved_docs)}  constraint-precision={precision:.2f}"
            + (f"  recall={len(expected_ids & retrieved_ids)}/{len(expected_ids)}" if case.check_recall else "")
            + (f"  answer-mentions-ok={mentions_ok}" if case.answer_must_mention else "")
            + f"  citations={len(resolved)}/{len(cited_refs)} resolve"
        )
        for doc in retrieved_docs:
            marker = "+" if case.predicate(doc.meta) else "-"
            print(f"    {marker} [doc {doc.id[:8]}] {doc.meta}")
    if usage:
        print(f"  tokens: { {k: v for k, v in usage.items() if isinstance(v, int)} }")
    for filters in filters_used:
        print(f"  filter: {filters}")
    print("  answer:")
    for line in answer.splitlines():
        print(f"    {line}")
    return {"passed": passed, "usage": usage, "time": elapsed}


# The small corpus: handcrafted documents with varied metadata (keyword, int, float, ISO date,
# language), small enough that every case's expected document set is enumerable.
SMALL_CORPUS = [
    Document(
        content="CRISPR-based gene editing was used to correct a hereditary blindness mutation in a clinical trial.",
        meta={"category": "science", "year": 2021, "rating": 4.6, "date": "2021-03-11", "language": "en"},
    ),
    Document(
        content="A quantum computer demonstrated error-corrected logical qubits outperforming physical qubits.",
        meta={"category": "science", "year": 2023, "rating": 4.8, "date": "2023-12-06", "language": "en"},
    ),
    Document(
        content="The LIGO observatory detected gravitational waves from two merging black holes for the first time.",
        meta={"category": "science", "year": 2016, "rating": 4.9, "date": "2016-02-11", "language": "en"},
    ),
    Document(
        content="Dolly the sheep became the first mammal cloned from an adult somatic cell.",
        meta={"category": "science", "year": 1996, "rating": 4.2, "date": "1996-07-05", "language": "en"},
    ),
    Document(
        content="The Berlin Wall fell, marking a decisive moment in the end of the Cold War.",
        meta={"category": "history", "year": 1989, "rating": 4.7, "date": "1989-11-09", "language": "en"},
    ),
    Document(
        content="The Apollo 11 mission landed the first humans on the Moon.",
        meta={"category": "history", "year": 1969, "rating": 5.0, "date": "1969-07-20", "language": "en"},
    ),
    Document(
        content="The Maastricht Treaty was signed, founding the European Union.",
        meta={"category": "history", "year": 1992, "rating": 3.9, "date": "1992-02-07", "language": "en"},
    ),
    Document(
        content="Leicester City won the Premier League despite 5000-1 preseason odds.",
        meta={"category": "sports", "year": 2016, "rating": 4.8, "date": "2016-05-02", "language": "en"},
    ),
    Document(
        content="Argentina won the FIFA World Cup final against France on penalties.",
        meta={"category": "sports", "year": 2022, "rating": 4.9, "date": "2022-12-18", "language": "en"},
    ),
    Document(
        content="Ein deutsches Team gewann die Champions League nach einem dramatischen Finale.",
        meta={"category": "sports", "year": 2013, "rating": 4.1, "date": "2013-05-25", "language": "de"},
    ),
]

# The large corpus: ~150k Amazon-Reviews-2023 reviews streamed from Hugging Face.
LARGE_CORPUS_CATEGORIES = ("All_Beauty", "Digital_Music", "Health_and_Personal_Care")
LARGE_CORPUS_DOCS_PER_CATEGORY = 50_000
_WRITE_BATCH_SIZE = 10_000


def build_document_store(backend: str, corpus: str) -> DocumentStore:
    """
    Build the (empty) document store for the chosen backend.

    :param backend: "in_memory" or "opensearch" (requires the `opensearch-haystack` package and a
        running OpenSearch, `OPENSEARCH_URL` or http://localhost:9200).
    :param corpus: The corpus name, used as part of the OpenSearch index name.
    :returns: The document store.
    """
    if backend == "opensearch":
        from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: PLC0415

        url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        return OpenSearchDocumentStore(
            hosts=url,
            index=f"advanced-rag-eval-{corpus}",
            # Credentials are read from OPENSEARCH_USERNAME / OPENSEARCH_PASSWORD by the store itself.
            use_ssl=url.startswith("https"),
            # Local docker instances use a self-signed certificate.
            verify_certs=not url.startswith("https://localhost"),
        )
    return InMemoryDocumentStore()


def build_retriever(store: DocumentStore):  # noqa: ANN201  (retriever type depends on the backend)
    """
    Build the matching BM25 retriever for the store.

    :param store: The document store.
    :returns: The retriever component.
    """
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=5)
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever  # noqa: PLC0415

    return OpenSearchBM25Retriever(document_store=store, top_k=5)


def populate_small_corpus(store: DocumentStore) -> None:
    """
    Write the small handcrafted corpus into the store.

    :param store: The document store to populate.
    """
    store.write_documents(SMALL_CORPUS, policy=DuplicatePolicy.OVERWRITE)


def _load_review_documents(category: str) -> list[Document]:
    """
    Stream one Amazon-Reviews-2023 category from Hugging Face and convert it to documents.

    :param category: The review category subset to load.
    :returns: Up to `LARGE_CORPUS_DOCS_PER_CATEGORY` documents with metadata for filtering.
    """
    from datasets import load_dataset  # noqa: PLC0415  (only needed for the `large` corpus)

    # The dataset repo is script-based (unsupported by datasets>=3), so stream its raw JSONL directly.
    dataset = load_dataset(
        "json",
        data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{category}.jsonl",
        split="train",
        streaming=True,
    )
    documents = []
    for row in itertools.islice(dataset, LARGE_CORPUS_DOCS_PER_CATEGORY):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        documents.append(
            Document(
                content=f"{row.get('title') or ''}. {text}"[:5_000],
                meta={
                    "category": category,
                    "rating": float(row["rating"]),
                    "helpful_vote": int(row["helpful_vote"]),
                    "verified_purchase": bool(row["verified_purchase"]),
                    "year": time.gmtime(row["timestamp"] / 1000).tm_year,
                    "asin": row["asin"],
                },
            )
        )
    return documents


def populate_large_corpus(store: DocumentStore) -> None:
    """
    Write the large review corpus into the store (streams from Hugging Face; requires `datasets`).

    :param store: The document store to populate.
    """
    for category in LARGE_CORPUS_CATEGORIES:
        documents = _load_review_documents(category)
        for batch_start in range(0, len(documents), _WRITE_BATCH_SIZE):
            store.write_documents(
                documents[batch_start : batch_start + _WRITE_BATCH_SIZE], policy=DuplicatePolicy.OVERWRITE
            )
        print(f"indexed {category}: {len(documents)} docs")


def main() -> None:
    """Run the selected eval set and print per-case reports plus a summary."""
    parser = argparse.ArgumentParser(description="Mini evaluation harness for the advanced RAG agent.")
    parser.add_argument("corpus", nargs="?", choices=("small", "large"), default="small")
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    args = parser.parse_args()
    corpus = args.corpus

    store = build_document_store(args.store, corpus)
    # A persistent store (OpenSearch) keeps its index across runs — skip re-indexing when populated.
    if store.count_documents() == 0:
        populate_small_corpus(store) if corpus == "small" else populate_large_corpus(store)
    else:
        print("store already populated, skipping indexing")
    print(f"corpus '{corpus}' on {args.store}: {store.count_documents()} docs")

    cases = SMALL_CASES if corpus == "small" else LARGE_CASES
    # The full id -> document map is only needed for recall (small corpus); a persistent store's
    # filter_documents() is capped (~10k on OpenSearch), so don't build it for the large corpus.
    documents_by_id = {doc.id: doc for doc in store.filter_documents()} if corpus == "small" else {}

    agent = create_advanced_rag_agent(document_store=store, retriever=build_retriever(store))

    results = [evaluate_case(agent, documents_by_id, case) for case in cases]

    passed = sum(r["passed"] for r in results)
    total_usage: dict[str, int] = {}
    for r in results:
        _sum_usage(total_usage, r["usage"])
    print(f"\n=== {passed}/{len(results)} cases passed ===")
    print(f"total time: {sum(r['time'] for r in results):.1f}s")
    if total_usage:
        print(f"total tokens: {total_usage}")
    if passed < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
