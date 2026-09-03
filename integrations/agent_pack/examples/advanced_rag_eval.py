# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate the Advanced RAG agent against shared small or large corpora.

The report covers retrieval correctness, metadata-inspection order, tool budgets, citations, answer requirements,
latency, and token usage. The small corpus has enumerable document-ID ground truth. The large corpus streams roughly
150,000 Amazon Reviews 2023 records and scores the metadata constraints satisfied by retrieved documents.

Run from `integrations/agent_pack` with `OPENAI_API_KEY` set:

    hatch run test:python examples/advanced_rag_eval.py small
    hatch run test:python examples/advanced_rag_eval.py large

The large corpus additionally requires `datasets`. Pass `--store opensearch` to reuse a persistent OpenSearch index;
set `OPENSEARCH_URL`, `OPENSEARCH_USERNAME`, and `OPENSEARCH_PASSWORD` when needed.
"""

import argparse
import time
from collections import Counter
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from util import LARGE_CASES, SMALL_CASES, EvalCase, build_document_store, build_retriever, populate_corpus

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    RETRIEVAL_TOOLS,
    extract_run_stats,
    score_advanced_rag_result,
)


def _sum_usage(total: dict[str, int], usage: dict[str, Any]) -> dict[str, int]:
    """Accumulate numeric token-usage entries into a running total."""
    for key, value in (usage or {}).items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
    return total


def evaluate_case(agent: Agent, documents_by_id: dict[str, Document], case: EvalCase) -> dict[str, Any]:
    """Run the Agent on one shared case and print its detailed report."""
    started = time.perf_counter()
    result = agent.run(messages=[ChatMessage.from_user(text=case.question)])
    elapsed = time.perf_counter() - started

    stats = extract_run_stats(messages=result["messages"])
    answer = result["last_message"].text or ""
    usage = result.get("token_usage") or {}

    labelled_documents = list(documents_by_id.values()) if case.check_recall else None
    evaluation_case = case.to_optimization_case(documents=labelled_documents)
    scored = score_advanced_rag_result(result=result, case=evaluation_case, latency_ms=elapsed * 1000)

    expected_ids = evaluation_case.expected_document_ids
    retrieved_docs = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_docs}
    cited_refs = scored.cited_document_ids
    resolved = [reference for reference in cited_refs if any(doc.id.startswith(reference) for doc in retrieved_docs)]
    within_budget = (
        stats.metadata_calls <= case.max_metadata_calls and stats.retrieval_calls <= case.max_retrieval_calls
    )

    counts = Counter(name for name, _ in stats.calls)
    filters_used = [args["filters"] for name, args in stats.calls if name in RETRIEVAL_TOOLS and args.get("filters")]
    print(f"\n[{'PASS' if scored.passed else 'FAIL'}] {case.question}")
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
        acknowledged = "answer_does_not_state_absence" not in scored.failures
        print(f"  expect-absent: acknowledged={acknowledged}")
    else:
        print(
            f"  retrieved={len(retrieved_docs)}  constraint-precision={scored.precision:.2f}"
            + (f"  recall={len(expected_ids & retrieved_ids)}/{len(expected_ids)}" if case.check_recall else "")
            + (f"  answer-mentions-ok={scored.answer_requirements_met}" if case.answer_must_mention else "")
            + f"  citations={len(resolved)}/{len(cited_refs)} resolve"
        )
        for document in retrieved_docs:
            marker = "+" if case.matches(document=document) else "-"
            print(f"    {marker} [doc {document.id[:8]}] {document.meta}")
    if scored.failures:
        print(f"  failures: {', '.join(scored.failures)}")
    if usage:
        print(f"  tokens: { {key: value for key, value in usage.items() if isinstance(value, int)} }")
    for filters in filters_used:
        print(f"  filter: {filters}")
    print("  answer:")
    for line in answer.splitlines():
        print(f"    {line}")
    return {"passed": scored.passed, "usage": usage, "time": elapsed}


def main() -> None:
    """Run the selected shared evaluation set and print per-case reports plus a summary."""
    parser = argparse.ArgumentParser(description="Evaluation harness for the Advanced RAG agent.")
    parser.add_argument("corpus", nargs="?", choices=("small", "large"), default="small")
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    arguments = parser.parse_args()

    store = build_document_store(backend=arguments.store, corpus=arguments.corpus)
    if store.count_documents() == 0:
        populate_corpus(store=store, corpus=arguments.corpus)
    else:
        print("store already populated, skipping indexing")
    print(f"corpus '{arguments.corpus}' on {arguments.store}: {store.count_documents()} docs")

    cases = SMALL_CASES if arguments.corpus == "small" else LARGE_CASES
    documents_by_id = (
        {document.id: document for document in store.filter_documents()} if arguments.corpus == "small" else {}
    )
    agent = create_advanced_rag_agent(document_store=store, retriever=build_retriever(store=store))
    results = [evaluate_case(agent=agent, documents_by_id=documents_by_id, case=case) for case in cases]

    passed = sum(result["passed"] for result in results)
    total_usage: dict[str, int] = {}
    for result in results:
        _sum_usage(total=total_usage, usage=result["usage"])
    print(f"\n=== {passed}/{len(results)} cases passed ===")
    print(f"total time: {sum(result['time'] for result in results):.1f}s")
    if total_usage:
        print(f"total tokens: {total_usage}")
    if passed < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
