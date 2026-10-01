# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Prepare the MultiHopRAG dataset as an evaluation set for harness optimization.
#
# The dataset pairs 609 news articles carrying real metadata — category, source, author, publication timestamp — with
# 2,556 labelled queries whose supporting evidence is quoted verbatim and attributed to its article.
#
# Articles are long — a median of 7,836 characters, up to 71,034. So articles are split by word with overlap.
#
# Splitting would normally blur the ground truth, since evidence is attributed to an article and not a chunk.
# So we ensure that a fact is contained wholly within a chunk. If a fact is split across two chunks, the query is
# excluded from the evaluation set.
#
# Run this module directly to build the corpus and report what it produced:
#
#     hatch run test:python examples/multihop_rag.py

import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.lazy_imports import LazyImport

with LazyImport(message='Run "pip install datasets" to build the MultiHopRAG evaluation set.') as datasets_import:
    from datasets import load_dataset  # type: ignore[import-untyped]

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


@dataclass
class LabelledQuestion:
    """
    One question the dataset labels, with the evidence an answer needs.

    :param question: The question as the dataset asks it.
    :param evidence: The quoted evidence the answer is built from, by the chunk it was found in.
    :param answer: The ground-truth answer, verbatim. How to check a reply against it is the harness's call: most
        are a name or a number, but some are short words like "Yes", where a substring check may match for
        reasons unrelated to the answer being right.
    """

    question: str
    evidence: dict[str, str]
    answer: str = ""

    @property
    def expected_document_ids(self) -> frozenset[str]:
        """The chunks an answer needs, which are the ones its evidence was found in."""
        return frozenset(self.evidence)


DATASET_ID = "yixuantt/MultiHopRAG"
CORPUS_KEY = "multihop-rag"

# Word-based splitting with overlap. Measured over the whole dataset at this setting: no evidence fact is split
# across a chunk boundary, so every one of the 6,084 is contained wholly within a chunk. However, the overlap
# means 926 of them are present in two adjacent chunks, so a query can no longer retrieve just a single expected
# document, so the query is dropped rather than scored loosely. So we end up with 1,432 of the 2,255 queries as
# valid eval points.
SPLIT_BY: Literal["word"] = "word"
SPLIT_LENGTH = 350
SPLIT_OVERLAP = 90

# Carried from the article onto each of its chunks
ARTICLE_METADATA = ("title", "category", "source", "author", "published_at", "url")

# Queries whose evidence is present in the corpus. The dataset's fourth type, `null_query`, is deliberately
# excluded for now to focus on answerable question types.
ANSWERABLE_QUESTION_TYPES = ("comparison_query", "inference_query", "temporal_query")


@dataclass
class Article:
    """One article's body and its chunks, in the order they appear in it."""

    body: str
    chunks: tuple[Document, ...]


def build_document_store(backend: str, index: str) -> DocumentStore:
    """
    Build an empty in-memory or OpenSearch document store under a named index.

    :param backend: Either "in_memory" or "opensearch".
    :param index: Names the index to open or create.
    :returns: The empty store.
    """
    if backend == "opensearch":
        opensearch_import.check()
        url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        return OpenSearchDocumentStore(
            hosts=url,
            index=index,
            use_ssl=url.startswith("https"),
            verify_certs=not url.startswith("https://localhost"),
        )
    return InMemoryDocumentStore(index=index)


def prepare_corpus(
    backend: Literal["in_memory", "opensearch"] = "in_memory", index: str = CORPUS_KEY
) -> tuple[DocumentStore, dict[str, Article]]:
    """
    Build the chunked corpus and index it, reusing an already-populated store.

    :param backend: Either "in_memory" or "opensearch".
    :param index: Names the index holding the corpus.
    :returns: The populated store and every article by title, each with the chunks covering it.
    """
    datasets_import.check()
    store = build_document_store(backend=backend, index=index)

    # Load the dataset, split the articles into chunks, and write them to the store.
    documents = [
        Document(content=row["body"], meta={field: row[field] for field in ARTICLE_METADATA})
        for row in load_dataset(DATASET_ID, "corpus", split="train")
        if (row["body"] or "").strip()
    ]
    splitter = DocumentSplitter(split_by=SPLIT_BY, split_length=SPLIT_LENGTH, split_overlap=SPLIT_OVERLAP)
    chunks = splitter.run(documents=documents)["documents"]

    # Write only when the store is empty or the corpus changed, so a repeated run does not reindex.
    if store.count_documents() != len(chunks):
        store.write_documents(documents=chunks, policy=DuplicatePolicy.OVERWRITE)

    # Group the chunks under the article they were split from, using the id the splitter records on each.
    by_source: dict[str, list[Document]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.meta["source_id"], []).append(chunk)
    articles = {
        document.meta["title"]: Article(
            body=document.content or "",
            chunks=tuple(sorted(by_source.get(document.id, []), key=lambda chunk: chunk.meta["split_idx_start"])),
        )
        for document in documents
    }
    return store, articles


def _covering_chunks(article: Article, fact: str) -> list[Document] | None:
    """
    Find the chunks a quoted evidence fact is contained in.

    A quote appearing more than once is ambiguous evidence, so if this occurs an empty list is returned to indicate
    that this datapoint should be excluded.

    :param article: The article the fact is attributed to.
    :param fact: The quoted evidence.
    :returns: The chunks holding every occurrence of the fact, or None when the quote is not in the article at
        all. An empty list when the quote is there but no single chunk holds all of it: it either spans a chunk boundary
        or is repeated in multiple chunks.
    """
    # Find all occurrences of the fact in the article.
    occurrences = []
    position = article.body.find(fact)
    while position != -1:
        occurrences.append(position)
        position = article.body.find(fact, position + 1)

    # If no occurrences we return None
    if not occurrences:
        return None

    # Return all chunks that cover all occurrences of the fact. If a fact is split across two chunks, an empty list is
    # returned.
    return [
        chunk
        for chunk in article.chunks
        if all(
            chunk.meta["split_idx_start"] <= start
            and start + len(fact) <= chunk.meta["split_idx_start"] + len(chunk.content or "")
            for start in occurrences
        )
    ]


def exact_eval_case_candidates(articles: dict[str, Article]) -> dict[str, list[LabelledQuestion]]:
    """
    Build every eval case whose expected documents can be stated exactly, grouped by question type.

    A query is kept when each of its evidence facts is contained in exactly one chunk. Overlap puts some facts in
    two adjacent chunks, and those queries are dropped.

    :param articles: The corpus the eval cases will be scored against, by article title.
    :returns: Eval cases per question type, ordered by question for reproducibility.
    :raises ValueError: If any evidence fact is split across chunks, since no single chunk then holds it.
    """
    datasets_import.check()
    candidates: dict[str, list[LabelledQuestion]] = {name: [] for name in ANSWERABLE_QUESTION_TYPES}
    split_across_chunks = 0
    for row in load_dataset(DATASET_ID, "MultiHopRAG", split="train"):
        if row["question_type"] not in candidates:
            continue

        # Locate every fact this query cites. One chunk per fact keeps the query; anything else drops it.
        evidence: dict[str, str] = {}
        exact = True
        for entry in row["evidence_list"]:
            fact = entry["fact"]
            holders = _covering_chunks(article=articles[entry["title"]], fact=fact)
            # holders is [] when the fact is in the article but split across chunks, or present in several
            split_across_chunks += holders == []
            if holders is None or len(holders) != 1:
                exact = False
                continue
            evidence[holders[0].id] = fact

        if exact and evidence:
            candidates[row["question_type"]].append(
                LabelledQuestion(question=row["query"], evidence=evidence, answer=row["answer"])
            )

    # The splitting settings decide this, so it is reachable by editing them.
    if split_across_chunks:
        msg = (
            f"{split_across_chunks} evidence facts are present but split across chunks, so no single chunk holds "
            f"them. Lower SPLIT_LENGTH or raise SPLIT_OVERLAP until every fact fits inside one chunk."
        )
        raise ValueError(msg)
    return {
        name: sorted(eval_cases, key=lambda eval_case: eval_case.question) for name, eval_cases in candidates.items()
    }


def build_eval_cases(articles: dict[str, Article], limit: int, seed: int = 0) -> list[LabelledQuestion]:
    """
    Build a reproducible set of eval cases, spread evenly across the question types.

    :param articles: The corpus the eval cases will be scored against, by article title.
    :param limit: How many eval cases to select. Each one costs an Agent run per candidate measured.
    :param seed: Seed for the selection, so the same evaluation set is rebuilt every time.
    :returns: The selected eval cases, interleaved by question type so a truncated set stays balanced.
    :raises ValueError: If `limit` is below one.
    """
    if limit < 1:
        msg = "limit must be at least 1."
        raise ValueError(msg)
    candidates = exact_eval_case_candidates(articles=articles)

    def rank(eval_case: LabelledQuestion) -> str:
        """Order eval cases by a hash of the seed and the question, which the same seed reproduces exactly."""
        return hashlib.sha256(f"{seed}:{eval_case.question}".encode()).hexdigest()

    # Sorted by a hash rather than seeded with `random`, whose ordering for a given seed can change between Python
    # versions.
    ordered = {name: sorted(eval_cases, key=rank) for name, eval_cases in candidates.items()}

    # Take one eval case per question type in turn, so stopping at the limit still leaves the types balanced.
    selected: list[LabelledQuestion] = []
    for position in range(max(len(pool) for pool in ordered.values())):
        for name in ANSWERABLE_QUESTION_TYPES:
            if position < len(ordered[name]):
                selected.append(ordered[name][position])
            if len(selected) == limit:
                return selected
    return selected


def _report(store: DocumentStore, articles: dict[str, Article], eval_cases: list[LabelledQuestion]) -> None:
    """Describe what was built, including the checks that make the eval cases scoreable."""
    chunks = [chunk for article in articles.values() for chunk in article.chunks]
    lengths = sorted(len(chunk.content or "") for chunk in chunks)
    metadata: dict[str, Any] = {field: {chunk.meta.get(field) for chunk in chunks} for field in ARTICLE_METADATA}
    print(f"  articles:  {len(articles)}")
    print(f"  chunks:    {len(chunks)} indexed={store.count_documents()} (mean {sum(lengths) // len(lengths)} chars)")
    print(
        f"  metadata:  category={len(metadata['category'])} source={len(metadata['source'])} "
        f"author={len(metadata['author'])} distinct values"
    )
    print(f"  published: {min(metadata['published_at'])} .. {max(metadata['published_at'])}")
    print(f"\n  eval cases selected: {len(eval_cases)}")
    # Listed rather than given as a range, since the sizes present need not be contiguous.
    sizes = sorted({len(eval_case.expected_document_ids) for eval_case in eval_cases})
    listed = str(sizes[0]) if len(sizes) == 1 else f"{', '.join(str(size) for size in sizes[:-1])} or {sizes[-1]}"
    print(f"    expected documents per eval case: {listed}")
    print(f"    with a ground-truth answer:  {sum(1 for eval_case in eval_cases if eval_case.answer)}")
    print("    example questions:")
    for eval_case in eval_cases[:3]:
        print(f"      - {eval_case.question[:96]}")


def main() -> None:
    """Build the corpus and evaluation set, and report what came out."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--max-eval-cases", type=int, default=20)
    parser.add_argument("--eval-case-seed", type=int, default=0)
    arguments = parser.parse_args()

    print(f"=== preparing {DATASET_ID} on {arguments.store} ===")
    store, articles = prepare_corpus(backend=arguments.store)
    eval_cases = build_eval_cases(articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed)
    _report(store=store, articles=articles, eval_cases=eval_cases)

    available = exact_eval_case_candidates(articles=articles)
    print(f"\n  eval cases available in total: {sum(len(pool) for pool in available.values())}")
    for name, pool in available.items():
        print(f"    {name:<20} {len(pool)}")


if __name__ == "__main__":
    main()
