# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Mini evaluation harness for the Advanced RAG agent, run against the MultiHopRAG corpus.
#
# The corpus and its labelled questions come from `multihop_rag`, which chunks the news articles and works out
# which chunks hold each question's quoted evidence. That gives exact ground truth: an eval case names the
# documents an answer needs, so retrieval is scored as recall over those IDs rather than a metadata predicate.
#
# Reported per eval case:
#
# - Retrieval: recall over the documents the question's evidence lives in, from the run's accumulated `documents`.
# - Process: whether metadata was inspected before any retrieval, per-tool call counts, how many retrievals used
#   a filter, tool errors, steps, and wall-clock time. An eval case fails when it exceeds its tool budget.
# - Citations: the answer must cite every document the question's evidence lives in, and every
#   `[doc <short-id>]` it carries must resolve to a document the run returned. Whether the answer is *right* is
#   not checked; that needs a judge this harness does not have yet.
# - Token usage, per eval case and totalled, for comparing models and reasoning efforts.
#
# Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:
#
#     hatch run test:python examples/advanced_rag_eval.py
#     hatch run test:python examples/advanced_rag_eval.py --max-eval-cases 5
#     hatch run test:python examples/advanced_rag_eval.py --store opensearch
#
# `--store opensearch` reuses a populated index across runs, so repeat runs skip re-indexing. Set `OPENSEARCH_URL`
# if it is not http://localhost:9200, and `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` for a secured instance.

import argparse
import re
import time
from collections import Counter
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets
from multihop_rag import CORPUS_KEY, LabelledQuestion, build_eval_cases, prepare_corpus
from util import build_bm25_retriever, preview

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.evaluation import (
    RAGEvalCase,
    ToolNames,
    ToolRunStats,
    budgets_exceeded,
    resolve_tool_budgets,
)

RETRIEVAL_TOOLS = ("search_documents", "fetch_documents_by_filter")
METADATA_TOOLS = ("list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range")
_CITATION_RE = re.compile(r"\[doc ([0-9a-f]{4,16})[^]]*\]")


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


def run_eval_case(agent: Agent, eval_case: RAGEvalCase, position: int, total: int) -> dict[str, Any]:
    """
    Run the agent on one eval case and print its report.

    :param agent: The agent under evaluation.
    :param eval_case: The eval case to evaluate.
    :param position: Which eval case this is, for the report heading.
    :param total: How many eval cases there are, for the report heading.
    :returns: A dict with `passed` (bool), `usage` (the run's token_usage dict), and `time` (s).
    """
    started = time.perf_counter()
    result = agent.run(messages=[ChatMessage.from_user(eval_case.question)])
    elapsed = time.perf_counter() - started

    # Everything the report needs comes out of the one run: its messages, its answer and its token usage.
    tool_run_stats = ToolRunStats.from_messages(messages=result["messages"])
    answer = result["last_message"].text or ""
    usage = result.get("token_usage") or {}

    # Recall on the retrieved documents: how many of the expected documents the Agent found.
    retrieved_docs = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_docs}
    found = eval_case.expected_document_ids & retrieved_ids
    recall = len(found) / len(eval_case.expected_document_ids)

    # Resolve each [doc <short-id>] the answer uses against what the Agent found. A reference matching nothing
    # is a fake citation, and one the answer doesn't use is an uncited document. Both are failures.
    cited_refs = _CITATION_RE.findall(answer)
    resolved = [ref for ref in cited_refs if any(document.id.startswith(ref) for document in retrieved_docs)]
    cited_ids = {document.id for document in retrieved_docs if any(document.id.startswith(r) for r in cited_refs)}
    citations_ok = len(resolved) == len(cited_refs)
    uncited = eval_case.expected_document_ids - cited_ids

    # An eval case passes only on all four: it found every expected document, cited every one of them, made no
    # citation that does not resolve, and stayed inside its tool budget.
    budgets = resolve_tool_budgets(
        budgets=eval_case.tool_budgets, tool_names=[tool.name for tool in flatten_tools_or_toolsets(tools=agent.tools)]
    )
    over_budget = budgets_exceeded(stats=tool_run_stats, budgets=budgets)
    passed = recall >= eval_case.min_recall and not uncited and citations_ok and not over_budget

    inspected_first = tool_run_stats.called_before(tools="list_metadata_fields", other=RETRIEVAL_TOOLS)
    counts = Counter(name for name, _ in tool_run_stats.calls)
    filtered_retrievals = tool_run_stats.calls_with_argument(tools=RETRIEVAL_TOOLS, argument="filters")
    needed = len(eval_case.expected_document_ids)
    print(f"\n=== eval case {position}/{total}: {'PASS' if passed else 'FAIL'} ===")
    print(f"  question: {eval_case.question}")
    print(f"  tools called: {dict(counts)}")
    print(
        f"  inspected metadata first: {inspected_first}   retrievals with a filter: {filtered_retrievals}   "
        f"tool errors: {len(tool_run_stats.errors)}   steps: {result['step_count']}   time: {elapsed:.1f}s"
    )
    for group, limit in budgets.items():
        used = tool_run_stats.calls_to(tools=group)
        print(f"  tool budget: {' + '.join(group)} {used}/{limit} -> {'ok' if used <= limit else 'EXCEEDED'}")
    print(f"  retrieval: found {len(found)}/{needed} of the documents the answer needs, {len(retrieved_docs)} returned")
    print(
        f"  citations: cited {needed - len(uncited)}/{needed} of them, "
        f"and {len(resolved)}/{len(cited_refs)} of the answer's references point at a returned document"
    )

    # Chunks of one article all carry its title, so they are grouped under it and told apart by chunk number.
    # Articles keep the order they were first returned in, which is the order the run ranked them.
    by_article: dict[str, list[Document]] = {}
    for document in retrieved_docs:
        by_article.setdefault(document.meta.get("title", ""), []).append(document)

    print("\n  documents returned, grouped by article. -> marks one the answer needs:")
    for title, documents in by_article.items():
        print(f"    Title: {title}")
        for document in sorted(documents, key=lambda chunk: chunk.meta.get("split_id", 0)):
            needed_here = "-> " if document.id in eval_case.expected_document_ids else "   "
            snippet = preview(text=document.content or "", limit=80)
            print(f"      {needed_here}chunk {document.meta.get('split_id'):>2}  [doc {document.id[:8]}]  {snippet}")
    # Naming the quote, since the id alone says nothing about what the run failed to find or failed to use.
    print()
    for document_id in sorted(eval_case.expected_document_ids - retrieved_ids):
        print(
            f"  needed but never retrieved: [doc {document_id[:8]}] "
            f"{preview(text=eval_case.evidence[document_id], limit=96)}"
        )
    for document_id in sorted(uncited & retrieved_ids):
        print(
            f"  needed and retrieved but not cited: [doc {document_id[:8]}] "
            f"{preview(text=eval_case.evidence[document_id], limit=96)}"
        )

    if usage:
        print(f"  tokens: { {k: v for k, v in usage.items() if isinstance(v, int)} }")
    for tool_name, message in tool_run_stats.errors:
        print(f"  tool error: {tool_name} -> {preview(text=message, limit=120)}")
    print("  answer:")
    for line in answer.splitlines():
        print(f"    {line}")
    return {"passed": passed, "usage": usage, "time": elapsed}


def main() -> None:
    """Run the eval set and print per-eval-case reports plus a summary."""
    parser = argparse.ArgumentParser(description="Mini evaluation harness for the Advanced RAG agent.")
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--max-eval-cases", type=int, default=10, help="How many labelled questions to evaluate.")
    parser.add_argument(
        "--eval-case-seed", type=int, default=0, help="Selects which eval cases are drawn from the dataset."
    )
    arguments = parser.parse_args()

    store, articles = prepare_corpus(backend=arguments.store)
    chunks = sum(len(article.chunks) for article in articles.values())
    print(f"{CORPUS_KEY} on {arguments.store}: {chunks} chunks from {len(articles)} articles")

    labelled: list[LabelledQuestion] = build_eval_cases(
        articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed
    )
    # Budgets for the tools this agent has. Lenient on purpose: too many retrievals is better than too few.
    # Generous on purpose: a MultiHopRAG question needs evidence from several articles, so several searches are
    # the expected shape of a good run rather than a sign of floundering.
    budgets: dict[ToolNames, int] = {METADATA_TOOLS: 8, RETRIEVAL_TOOLS: 12}
    eval_cases = [
        RAGEvalCase(question=question.question, evidence=question.evidence, tool_budgets=budgets)
        for question in labelled
    ]
    print(f"eval cases: {len(eval_cases)} labelled from evidence")

    # Build the advanced rag agent
    agent = create_advanced_rag_agent(document_store=store, retriever=build_bm25_retriever(store=store))

    # Run the eval cases
    results = [
        run_eval_case(agent=agent, eval_case=eval_case, position=position, total=len(eval_cases))
        for position, eval_case in enumerate(eval_cases, start=1)
    ]

    # Calculate total usage
    total_usage: dict[str, int] = {}
    for result in results:
        _sum_usage(total=total_usage, usage=result["usage"])

    # Print the results
    passed = sum(result["passed"] for result in results)
    print(f"\n=== {passed}/{len(results)} eval cases passed ===")
    print(f"total time: {sum(result['time'] for result in results):.1f}s")
    if total_usage:
        print(f"total tokens: {total_usage}")


if __name__ == "__main__":
    main()
