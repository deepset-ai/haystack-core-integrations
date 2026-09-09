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


@dataclass(frozen=True)
class LabelledQuestion:
    """
    One question the dataset labels, with the evidence an answer needs.

    :param question: The question as the dataset asks it.
    :param expected_document_ids: The chunks holding the evidence the answer is built from.
    :param answer_must_mention: The ground-truth answer, when a substring check on it means anything.
    """

    question: str
    expected_document_ids: frozenset[str]
    answer_must_mention: tuple[str, ...] = ()


with LazyImport(message='Run "pip install datasets" to build the MultiHopRAG evaluation set.') as datasets_import:
    from datasets import load_dataset

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

DATASET_ID = "yixuantt/MultiHopRAG"
CORPUS_KEY = "multihop-rag"

# Word-based splitting with overlap. Measured over the whole dataset at this setting: no evidence fact is split
# across a chunk boundary, so every one of the 6,084 is contained wholly within a chunk. Overlap is what costs:
# 926 of them sit in the region two adjacent chunks share, and a query touching one of those cannot name a single
# expected document, so it is dropped rather than scored loosely. 1,432 of the 2,255 queries survive that.
SPLIT_BY = "word"
SPLIT_LENGTH = 350
SPLIT_OVERLAP = 90

# Carried from the article onto each of its chunks, so the metadata tools see the same fields whichever chunk is
# retrieved.
ARTICLE_METADATA = ("title", "category", "source", "author", "published_at", "url")

# Queries whose evidence is present in the corpus. The dataset's fourth type, `null_query`, is deliberately
# excluded: its answer is "Insufficient information." while topically related articles do exist, so scoring it
# needs an expectation this harness does not have yet — retrieval is appropriate, but the answer must decline.
ANSWERABLE_QUESTION_TYPES = ("comparison_query", "inference_query", "temporal_query")

# A ground-truth answer is asserted only when a substring check means something. Most answers are a name or a
# number, but many comparison answers are "Yes" or "No", where finding the word in a sentence proves nothing.
_UNASSERTABLE_ANSWERS = frozenset({"yes", "no", "true", "false", "insufficient information."})
_MIN_ASSERTABLE_ANSWER_CHARS = 5


@dataclass(frozen=True)
class Article:
    """
    One article's body alongside the chunks covering it, ordered by where each chunk starts.

    An evidence quote is a span of the body, so which chunks hold it follows from arithmetic on the offsets
    `DocumentSplitter` records rather than from searching each chunk for the text.
    """

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

    # Grouped on the id the splitter records, rather than on the title, so the mapping holds whatever the titles do.
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

    The quote is located in the article first, and every place it occurs is located, not just the first. A quote
    appearing twice is ambiguous evidence, and resolving it to whichever copy came first would silently attach
    the case to an arbitrary one of them.

    :param article: The article the fact is attributed to.
    :param fact: The quoted evidence.
    :returns: The chunks containing every occurrence of the fact, or None when the quote is not in the article at
        all. A quote that occurs but spans a chunk boundary is contained in no chunk, and returns an empty list.
    """
    occurrences = []
    position = article.body.find(fact)
    while position != -1:
        occurrences.append(position)
        position = article.body.find(fact, position + 1)
    if not occurrences:
        return None
    return [
        chunk
        for chunk in article.chunks
        if all(
            chunk.meta["split_idx_start"] <= start
            and start + len(fact) <= chunk.meta["split_idx_start"] + len(chunk.content or "")
            for start in occurrences
        )
    ]


def _answer_terms(answer: str) -> tuple[str, ...]:
    """Return the ground-truth answer as an assertable term, or nothing when a substring check proves nothing."""
    cleaned = (answer or "").strip()
    if len(cleaned) < _MIN_ASSERTABLE_ANSWER_CHARS or cleaned.lower() in _UNASSERTABLE_ANSWERS:
        return ()
    return (cleaned,)


def exact_eval_case_candidates(articles: dict[str, Article]) -> dict[str, list[LabelledQuestion]]:
    """
    Build every eval case whose expected documents can be stated exactly, grouped by question type.

    A query qualifies when each of its evidence facts is contained in exactly one chunk. Facts sitting in two
    adjacent chunks are the price of overlap, and a case naming both would fail an Agent that retrieved either,
    so those queries are left out rather than scored loosely.

    :param articles: The corpus the eval cases will be scored against, by article title.
    :returns: Eval cases per question type, ordered by question for reproducibility.
    :raises ValueError: If any evidence fact is missing from the article it is attributed to, or is present but
        split across chunks, since either means the corpus and the labels no longer correspond.
    """
    datasets_import.check()
    candidates: dict[str, list[LabelledQuestion]] = {name: [] for name in ANSWERABLE_QUESTION_TYPES}
    missing = 0
    split_across_chunks = 0
    for row in load_dataset(DATASET_ID, "MultiHopRAG", split="train"):
        if row["question_type"] not in candidates:
            continue
        expected: set[str] = set()
        exact = bool(row["evidence_list"])
        for entry in row["evidence_list"]:
            article = articles.get(entry["title"])
            holders = (
                None if article is None else _covering_chunks(article=article, fact=(entry.get("fact") or "").strip())
            )
            missing += holders is None
            split_across_chunks += holders == []
            if holders is None or len(holders) != 1:
                exact = False
                continue
            expected.add(holders[0].id)
        if exact and expected:
            candidates[row["question_type"]].append(
                LabelledQuestion(
                    question=row["query"],
                    expected_document_ids=frozenset(expected),
                    answer_must_mention=_answer_terms(answer=row["answer"]),
                )
            )
    if missing:
        msg = (
            f"{missing} evidence facts are not present in the article they are attributed to, so eval cases "
            f"cannot be scored against this corpus. The dataset changed."
        )
        raise ValueError(msg)
    if split_across_chunks:
        msg = (
            f"{split_across_chunks} evidence facts are present but split across chunks, so no single chunk holds "
            f"them. Lower SPLIT_LENGTH or raise SPLIT_OVERLAP until every fact fits inside one chunk."
        )
        raise ValueError(msg)
    return {name: sorted(cases, key=lambda case: case.question) for name, cases in candidates.items()}


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

    def rank(case: LabelledQuestion) -> str:
        """Order eval cases by a hash of the seed and the question."""
        return hashlib.sha256(f"{seed}:{case.question}".encode()).hexdigest()

    # Hashed rather than shuffled: the same seed has to rebuild the same evaluation set, and the ordering `random`
    # produces for a given seed is not guaranteed to hold across Python versions.
    ordered = {name: sorted(cases, key=rank) for name, cases in candidates.items()}

    selected: list[LabelledQuestion] = []
    for position in range(max(len(pool) for pool in ordered.values())):
        for name in ANSWERABLE_QUESTION_TYPES:
            if position < len(ordered[name]):
                selected.append(ordered[name][position])
            if len(selected) == limit:
                return selected
    return selected


def _report(store: DocumentStore, articles: dict[str, Article], cases: list[LabelledQuestion]) -> None:
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
    print(f"\n  cases selected: {len(cases)}")
    print(f"    expected documents per eval case: {sorted({len(case.expected_document_ids) for case in cases})}")
    print(f"    with an assertable answer:   {sum(1 for case in cases if case.answer_must_mention)}")
    for case in cases[:3]:
        print(f"    - {case.question[:96]}")


def main() -> None:
    """Build the corpus and evaluation set, and report what came out."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--case-seed", type=int, default=0)
    arguments = parser.parse_args()

    print(f"=== preparing {DATASET_ID} on {arguments.store} ===")
    store, articles = prepare_corpus(backend=arguments.store)
    cases = build_eval_cases(articles=articles, limit=arguments.max_cases, seed=arguments.case_seed)
    _report(store=store, articles=articles, cases=cases)

    available = exact_eval_case_candidates(articles=articles)
    print(f"\n  cases available in total: {sum(len(pool) for pool in available.values())}")
    for name, pool in available.items():
        print(f"    {name:<20} {len(pool)}")


if __name__ == "__main__":
    main()
