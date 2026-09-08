# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Prepare the MultiHopRAG dataset as an evaluation set for harness optimization.
#
# The dataset pairs 609 news articles carrying real metadata — category, source, author, publication timestamp — with
# 2,556 labelled queries whose supporting evidence is quoted verbatim and attributed to its article.
#
# Articles are long — a median of 7,836 characters, up to 71,034 — and the retrieval tools print a retrieved
# document's content in full. Left whole, one retrieval would put tens of thousands of characters into the Agent's
# context. So articles are split. Splitting is by word with overlap.
#
# Splitting would normally blur the ground truth, since evidence is attributed to an article and retrieval then
# returns pieces of one. It does not here: every one of the 6,084 evidence facts can be located in its article, and
# at `SPLIT_LENGTH`/`SPLIT_OVERLAP` every one of them is contained wholly within a chunk. So a case's expected
# documents are computed from which chunks contain its evidence. `build_eval_cases` keeps only the queries whose
# every fact is contained within exactly one chunk, which leaves the expectation exact: overlap means some facts
# sit in two adjacent chunks, and demanding both would fail an Agent that retrieved either.
#
# Run this module directly to build the corpus and report what it produced, including the checks that the mapping
# still holds:
#
#     hatch run test:python examples/multihop_rag.py

import argparse
import hashlib
from typing import Any, Literal

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.lazy_imports import LazyImport
from util import build_document_store

from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase

with LazyImport(message='Run "pip install datasets" to build the MultiHopRAG evaluation set.') as datasets_import:
    from datasets import load_dataset

DATASET_ID = "yixuantt/MultiHopRAG"
CORPUS_KEY = "multihop-rag"

# Word-based splitting with overlap. Measured over the whole dataset: every evidence fact is contained wholly
# within a chunk at this setting, which is what keeps a case's expected documents exact. Chunks average ~2,000
# characters, which is also about as much as is reasonable to put in the Agent's context per retrieved document.
SPLIT_BY = "word"
SPLIT_LENGTH = 350
SPLIT_OVERLAP = 90

# Carried from the article onto each of its chunks, so the metadata tools see the same fields whichever chunk is
# retrieved. `published_at` is a zero-padded "YYYY-MM-DD HH:MM:SS" string, which orders chronologically under a
# plain string comparison, so range filters and `get_metadata_field_range` work on it directly.
ARTICLE_METADATA = ("title", "category", "source", "author", "published_at", "url")

# Queries whose evidence is present in the corpus. The dataset's fourth type, `null_query`, is deliberately
# excluded: its answer is "Insufficient information." while topically related articles do exist, so scoring it
# needs an expectation this harness does not have yet — retrieval is appropriate, but the answer must decline.
ANSWERABLE_QUESTION_TYPES = ("comparison_query", "inference_query", "temporal_query")

# A ground-truth answer is asserted only when a substring check means something. Most answers are a name or a
# number, but many comparison answers are "Yes" or "No", where finding the word in a sentence proves nothing.
_UNASSERTABLE_ANSWERS = frozenset({"yes", "no", "true", "false", "insufficient information."})
_MIN_ASSERTABLE_ANSWER_CHARS = 5


def prepare_corpus(
    backend: Literal["in_memory", "opensearch"] = "in_memory", index: str = CORPUS_KEY
) -> tuple[DocumentStore, list[Document]]:
    """
    Build the chunked corpus and index it, reusing an already-populated store.

    :param backend: Either "in_memory" or "opensearch".
    :param index: Names the index holding the corpus.
    :returns: The populated store and the chunks it holds.
    """
    datasets_import.check()
    store = build_document_store(backend=backend, index=index)
    # The body alone becomes the content: the title travels in metadata, where the retrieval tools print it
    # alongside every chunk, and keeping the content untouched is what makes the measured evidence mapping hold.
    articles = [
        Document(content=row["body"], meta={field: row[field] for field in ARTICLE_METADATA})
        for row in load_dataset(DATASET_ID, "corpus", split="train")
        if (row["body"] or "").strip()
    ]
    splitter = DocumentSplitter(split_by=SPLIT_BY, split_length=SPLIT_LENGTH, split_overlap=SPLIT_OVERLAP)
    chunks = splitter.run(documents=articles)["documents"]
    if store.count_documents() != len(chunks):
        store.write_documents(documents=chunks, policy=DuplicatePolicy.OVERWRITE)
    return store, chunks


def _chunks_by_article(chunks: list[Document]) -> dict[str, list[Document]]:
    """Group chunks under their article title."""
    grouped: dict[str, list[Document]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.meta["title"], []).append(chunk)
    return grouped


def _containing_chunks(chunks: list[Document], fact: str) -> list[Document]:
    """
    Find the chunks holding one evidence fact.

    :param chunks: The candidate chunks, all from the article the fact is attributed to.
    :param fact: The quoted evidence.
    :returns: Every chunk containing the fact, matched on its opening words when the quote was lightly reworded.
    """
    if matched := [chunk for chunk in chunks if fact in (chunk.content or "")]:
        return matched
    opening = " ".join(fact.split()[:8])
    return [chunk for chunk in chunks if opening and opening in (chunk.content or "")]


def _answer_terms(answer: str) -> tuple[str, ...]:
    """Return the ground-truth answer as an assertable term, or nothing when a substring check proves nothing."""
    cleaned = (answer or "").strip()
    if len(cleaned) < _MIN_ASSERTABLE_ANSWER_CHARS or cleaned.lower() in _UNASSERTABLE_ANSWERS:
        return ()
    return (cleaned,)


def exact_case_candidates(chunks: list[Document]) -> dict[str, list[AdvancedRAGEvaluationCase]]:
    """
    Build every eval case whose expected documents can be stated exactly, grouped by question type.

    A query qualifies when each of its evidence facts is contained in exactly one chunk. Facts sitting in two
    chunks are the price of overlap, and a case naming both would fail an Agent that retrieved either, so those
    queries are left out rather than scored loosely.

    :param chunks: The chunked corpus the eval cases will be scored against.
    :returns: Eval cases per question type, ordered by question for reproducibility.
    :raises ValueError: If any evidence fact cannot be located at all, which would mean the corpus and the labels
        no longer correspond.
    """
    datasets_import.check()
    grouped = _chunks_by_article(chunks=chunks)
    candidates: dict[str, list[AdvancedRAGEvaluationCase]] = {name: [] for name in ANSWERABLE_QUESTION_TYPES}
    unlocatable = 0
    for row in load_dataset(DATASET_ID, "MultiHopRAG", split="train"):
        if row["question_type"] not in candidates:
            continue
        expected: set[str] = set()
        exact = bool(row["evidence_list"])
        for entry in row["evidence_list"]:
            holders = _containing_chunks(chunks=grouped.get(entry["title"], []), fact=(entry.get("fact") or "").strip())
            unlocatable += not holders
            if len(holders) != 1:
                exact = False
                continue
            expected.add(holders[0].id)
        if exact and expected:
            candidates[row["question_type"]].append(
                AdvancedRAGEvaluationCase(
                    question=row["query"],
                    expected_document_ids=frozenset(expected),
                    answer_must_mention=_answer_terms(answer=row["answer"]),
                    require_metadata_inspection=False,
                )
            )
    if unlocatable:
        msg = (
            f"{unlocatable} evidence facts could not be found in the corpus, so eval cases cannot be scored "
            f"against it. The dataset or the splitting settings changed."
        )
        raise ValueError(msg)
    return {name: sorted(cases, key=lambda case: case.question) for name, cases in candidates.items()}


def build_eval_cases(chunks: list[Document], limit: int, seed: int = 0) -> list[AdvancedRAGEvaluationCase]:
    """
    Build a reproducible set of eval cases, spread evenly across the question types.

    :param chunks: The chunked corpus the eval cases will be scored against.
    :param limit: How many eval cases to select. Each one costs an Agent run per candidate measured.
    :param seed: Seed for the selection, so the same evaluation set is rebuilt every time.
    :returns: The selected eval cases, interleaved by question type so a truncated set stays balanced.
    :raises ValueError: If `limit` is below one.
    """
    if limit < 1:
        msg = "limit must be at least 1."
        raise ValueError(msg)
    candidates = exact_case_candidates(chunks=chunks)

    def rank(case: AdvancedRAGEvaluationCase) -> str:
        """Order eval cases by a hash of the seed and the question."""
        return hashlib.sha256(f"{seed}:{case.question}".encode()).hexdigest()

    # Hashed rather than shuffled: the same seed has to rebuild the same evaluation set, and the ordering `random`
    # produces for a given seed is not guaranteed to hold across Python versions.
    ordered = {name: sorted(cases, key=rank) for name, cases in candidates.items()}

    selected: list[AdvancedRAGEvaluationCase] = []
    for position in range(max(len(pool) for pool in ordered.values())):
        for name in ANSWERABLE_QUESTION_TYPES:
            if position < len(ordered[name]):
                selected.append(ordered[name][position])
            if len(selected) == limit:
                return selected
    return selected


def _report(store: DocumentStore, chunks: list[Document], cases: list[AdvancedRAGEvaluationCase]) -> None:
    """Describe what was built, including the checks that make the eval cases scoreable."""
    articles = {chunk.meta["title"] for chunk in chunks}
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
    print(f"    expected documents per case: {sorted({len(case.expected_document_ids) for case in cases})}")
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
    store, chunks = prepare_corpus(backend=arguments.store)
    cases = build_eval_cases(chunks=chunks, limit=arguments.max_cases, seed=arguments.case_seed)
    _report(store=store, chunks=chunks, cases=cases)

    available = exact_case_candidates(chunks=chunks)
    print(f"\n  cases available in total: {sum(len(pool) for pool in available.values())}")
    for name, pool in available.items():
        print(f"    {name:<20} {len(pool)}")


if __name__ == "__main__":
    main()
