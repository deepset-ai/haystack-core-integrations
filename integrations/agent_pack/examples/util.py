# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared corpora, cases, and document-store helpers for the Advanced RAG examples."""

import itertools
import os
import time
from dataclasses import dataclass
from typing import Any

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils.filters import document_matches_filter

from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase

LARGE_CORPUS_CATEGORIES = ("All_Beauty", "Digital_Music", "Health_and_Personal_Care")
LARGE_CORPUS_DOCS_PER_CATEGORY = 50_000
_WRITE_BATCH_SIZE = 10_000


def _comparison(field: str, operator: str, value: Any) -> dict[str, Any]:
    """Build one Haystack metadata comparison filter."""
    return {"field": f"meta.{field}", "operator": operator, "value": value}


def _and(*conditions: dict[str, Any]) -> dict[str, Any]:
    """Combine Haystack metadata filters with a logical AND."""
    return {"operator": "AND", "conditions": list(conditions)}


@dataclass(frozen=True)
class EvalCase:
    """One shared evaluation case expressed with serializable metadata ground truth."""

    question: str
    filters: dict[str, Any] | None
    check_recall: bool = True
    min_docs: int = 1
    answer_must_mention: tuple[str, ...] = ()
    min_precision: float = 0.5
    expect_absent: bool = False
    max_metadata_calls: int = 5
    max_retrieval_calls: int = 5

    def matches(self, document: Document) -> bool:
        """Return whether a document satisfies this case's metadata ground truth."""
        return self.filters is not None and document_matches_filter(filters=self.filters, document=document)

    def to_optimization_case(self, documents: list[Document] | None = None) -> AdvancedRAGEvaluationCase:
        """Convert this definition to the optimizer evaluator's serializable case model."""
        common = {
            "question": self.question,
            "answer_must_mention": self.answer_must_mention,
            "max_metadata_calls": self.max_metadata_calls,
            "max_retrieval_calls": self.max_retrieval_calls,
        }
        if self.expect_absent:
            return AdvancedRAGEvaluationCase(expect_absent=True, **common)
        if self.check_recall:
            expected_ids = frozenset(document.id for document in documents or [] if self.matches(document=document))
            return AdvancedRAGEvaluationCase(expected_document_ids=expected_ids, **common)
        return AdvancedRAGEvaluationCase(
            expected_metadata_filter=self.filters,
            min_matching_documents=self.min_docs,
            min_precision=self.min_precision,
            **common,
        )


SMALL_CASES = [
    EvalCase(
        question="What scientific breakthroughs happened after 2015, according to the documents?",
        filters=_and(_comparison("category", "==", "science"), _comparison("year", ">", 2015)),
        answer_must_mention=("quantum", "CRISPR"),
    ),
    EvalCase(
        question="Which historical events in the documents happened before 1990?",
        filters=_and(_comparison("category", "==", "history"), _comparison("year", "<", 1990)),
        answer_must_mention=("Berlin", "Apollo"),
    ),
    EvalCase(
        question="What does the German-language document describe?",
        filters=_comparison("language", "==", "de"),
        answer_must_mention=("Champions League",),
    ),
    EvalCase(
        question="What does the single highest-rated document describe?",
        filters=_comparison("rating", "==", 5.0),
        answer_must_mention=("Apollo",),
    ),
    EvalCase(
        question="What do the documents say about sports events from 2020 onwards?",
        filters=_and(_comparison("category", "==", "sports"), _comparison("year", ">=", 2020)),
        answer_must_mention=("Argentina",),
    ),
    EvalCase(
        question="What is CRISPR used for according to the documents?",
        filters=_and(_comparison("category", "==", "science"), _comparison("year", "==", 2021)),
        answer_must_mention=("blindness",),
    ),
    EvalCase(
        question="What do the documents in the 'food' category say about cooking?",
        filters=None,
        expect_absent=True,
    ),
    EvalCase(question="Which of the documents are written in French?", filters=None, expect_absent=True),
]

LARGE_CASES = [
    EvalCase(
        question=(
            "What do verified purchasers complain about in low-rated (1-2 star) beauty product reviews from 2022 "
            "or later?"
        ),
        filters=_and(
            _comparison("category", "==", "All_Beauty"),
            _comparison("rating", "<=", 2.0),
            _comparison("year", ">=", 2022),
            _comparison("verified_purchase", "==", True),
        ),
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="What did reviewers think of digital music purchases before 2010?",
        filters=_and(_comparison("category", "==", "Digital_Music"), _comparison("year", "<", 2010)),
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question=(
            "List every 1-star health product review with 10 or more helpful votes, and say what each one "
            "complains about."
        ),
        filters=_and(
            _comparison("category", "==", "Health_and_Personal_Care"),
            _comparison("rating", "==", 1.0),
            _comparison("helpful_vote", ">=", 10),
        ),
        # Recall-checked on purpose. The answer has to account for the complete filtered set, so a configuration
        # that can only surface a document or two per retrieval fails here even when it passes the summary cases
        # above, which any single adequate retrieval satisfies. The retrieval budget is deliberately too small to
        # page a starved fetch limit around the requirement: a configuration sized for this set reaches it in one
        # or two calls, while one capped at a couple of documents per fetch needs far more than three.
        check_recall=True,
        max_retrieval_calls=3,
    ),
    EvalCase(
        question="Summarize what the most helpful health product reviews (10 or more helpful votes) say.",
        filters=_and(
            _comparison("category", "==", "Health_and_Personal_Care"),
            _comparison("helpful_vote", ">=", 10),
        ),
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="What are common themes in 5-star beauty product reviews from 2020?",
        filters=_and(
            _comparison("category", "==", "All_Beauty"),
            _comparison("rating", "==", 5.0),
            _comparison("year", "==", 2020),
        ),
        check_recall=False,
        min_docs=3,
    ),
    EvalCase(
        question="What do reviews in the Electronics category say about laptop battery life?",
        filters=None,
        check_recall=False,
        expect_absent=True,
    ),
    EvalCase(
        question="What do beauty product reviews from 2030 or later say?",
        filters=None,
        check_recall=False,
        expect_absent=True,
    ),
]

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


def build_document_store(backend: str, corpus: str) -> DocumentStore:
    """Build an empty in-memory or OpenSearch document store for a corpus."""
    if backend == "opensearch":
        from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: PLC0415

        url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        return OpenSearchDocumentStore(
            hosts=url,
            index=f"advanced-rag-eval-{corpus}",
            use_ssl=url.startswith("https"),
            verify_certs=not url.startswith("https://localhost"),
        )
    return InMemoryDocumentStore()


def build_retriever(store: DocumentStore, top_k: int = 5):  # noqa: ANN201
    """Build the matching BM25 retriever for a document store."""
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever  # noqa: PLC0415

    return OpenSearchBM25Retriever(document_store=store, top_k=top_k)


def populate_small_corpus(store: DocumentStore) -> None:
    """Write the small handcrafted corpus into a document store."""
    store.write_documents(documents=SMALL_CORPUS, policy=DuplicatePolicy.OVERWRITE)


def _load_review_documents(category: str, documents_per_category: int) -> list[Document]:
    """Stream one Amazon Reviews 2023 category from Hugging Face and convert it to documents."""
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(
        "json",
        data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{category}.jsonl",
        split="train",
        streaming=True,
    )
    documents = []
    for row in itertools.islice(dataset, documents_per_category):
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


def populate_large_corpus(store: DocumentStore, documents_per_category: int = LARGE_CORPUS_DOCS_PER_CATEGORY) -> None:
    """Stream and write the large Amazon Reviews 2023 corpus into a document store."""
    for category in LARGE_CORPUS_CATEGORIES:
        documents = _load_review_documents(category=category, documents_per_category=documents_per_category)
        for batch_start in range(0, len(documents), _WRITE_BATCH_SIZE):
            store.write_documents(
                documents=documents[batch_start : batch_start + _WRITE_BATCH_SIZE],
                policy=DuplicatePolicy.OVERWRITE,
            )
        print(f"indexed {category}: {len(documents)} docs")


def populate_corpus(
    store: DocumentStore, corpus: str, documents_per_category: int = LARGE_CORPUS_DOCS_PER_CATEGORY
) -> None:
    """Populate an empty document store with the selected shared corpus."""
    if corpus == "small":
        populate_small_corpus(store=store)
    else:
        populate_large_corpus(store=store, documents_per_category=documents_per_category)
