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
#     hatch run test:python examples/advanced_rag_eval.py --max-cases 5
#     hatch run test:python examples/advanced_rag_eval.py --store opensearch
#
# `--store opensearch` reuses a populated index across runs, so repeat runs skip re-indexing. Set `OPENSEARCH_URL`
# if it is not http://localhost:9200, and `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` for a secured instance.

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from haystack.components.agents import Agent
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.lazy_imports import LazyImport
from multihop_rag import CORPUS_KEY, LabelledQuestion, build_eval_cases, prepare_corpus

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever

RETRIEVAL_TOOLS = ("search_documents", "fetch_documents_by_filter")
METADATA_TOOLS = ("list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range")
_CITATION_RE = re.compile(r"\[doc ([0-9a-f]{4,16})\]")


@dataclass
class EvalCase:
    """
    One eval case: a question, the documents that answer it, and the budgets the run may spend.

    :param question: The question to put to the agent.
    :param evidence: Ground truth, as `{chunk id: the quote found in that chunk}`. The keys are the documents the
        answer needs, and the values are what the answer should be based on. For example,
        {"a1b2c3...": "Tyreek Hill now needs to ...", "d4e5f6...": "The Dolphins went on to ..."}
    :param tool_budgets: How many times the run may call a tool, or a group of tools sharing one allowance, as
        `{tool name or names: limit}`. Exceeding any of them fails the eval case. Which tools these are depends
        on the agent under evaluation, so the caller supplies them.
    """

    question: str
    evidence: dict[str, str]
    tool_budgets: dict[str | tuple[str, ...], int]

    @property
    def expected_document_ids(self) -> frozenset[str]:
        """The chunks an answer needs, which are the ones its evidence was found in."""
        return frozenset(self.evidence)


@dataclass
class ToolRunStats:
    """
    The tool calls one agent run made, and what they add up to.

    :param calls: Every call the run made, in the order it made them, as `(tool name, the arguments it passed)`:

            [("list_metadata_fields", {}), ("search_documents", {"query": "CRISPR", "filters": None})]

    :param errors: The calls that came back an error, as `(tool name, what it said)`:

            [("get_metadata_field_values", "field 'nope' does not exist in the store")]
    """

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @staticmethod
    def _named(tools: str | tuple[str, ...]) -> set[str]:
        """
        Normalize one tool name or a group of them to a set.

        :param tools: A tool name, or several of them.
        :returns: The names as a set.
        """
        return {tools} if isinstance(tools, str) else set(tools)

    def calls_to(self, tools: str | tuple[str, ...]) -> int:
        """
        Count the calls made to one tool, or to any of a group of them.

        :param tools: A tool name, or several of them.
        :returns: How many calls the run made to them.
        """
        wanted = self._named(tools=tools)
        return sum(1 for name, _ in self.calls if name in wanted)

    def called_before(self, tools: str | tuple[str, ...], other: str | tuple[str, ...]) -> bool:
        """
        Whether the run reached for one tool before it reached for another.

        :param tools: The tool, or tools, that should come first.
        :param other: The tool, or tools, they should come before.
        :returns: True when one of `tools` was called and none of `other` was called before it. False when
            `other` came first, and when neither was called at all.
        """
        wanted, after = self._named(tools=tools), self._named(tools=other)
        for name, _ in self.calls:
            if name in wanted:
                return True
            if name in after:
                return False
        return False


def extract_tool_run_stats(messages: list[ChatMessage]) -> ToolRunStats:
    """
    Extract tool calls and error results from an agent run.

    :param messages: The messages returned by `agent.run(...)`.
    :returns: The extracted statistics.
    """
    return ToolRunStats(
        calls=[(call.tool_name, call.arguments or {}) for message in messages for call in message.tool_calls],
        errors=[
            (result.origin.tool_name, result.result)
            for message in messages
            for result in message.tool_call_results
            if result.error
        ],
    )


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


def build_bm25_retriever(store: DocumentStore, top_k: int = 5):  # noqa: ANN201
    """
    Build the matching BM25 retriever for a document store.

    :param store: The store to retrieve from.
    :param top_k: How many documents one retrieval returns.
    :returns: The retriever.
    """
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    opensearch_import.check()
    return OpenSearchBM25Retriever(document_store=store, top_k=top_k)


def run_eval_case(agent: Agent, case: EvalCase, position: int, total: int) -> dict[str, Any]:
    """
    Run the agent on one eval case and print its report.

    :param agent: The agent under evaluation.
    :param case: The eval case to evaluate.
    :param position: Which eval case this is, for the report heading.
    :param total: How many eval cases there are, for the report heading.
    :returns: A dict with `passed` (bool), `usage` (the run's token_usage dict), and `time` (s).
    """
    started = time.perf_counter()
    result = agent.run(messages=[ChatMessage.from_user(case.question)])
    elapsed = time.perf_counter() - started

    # Everything the report needs comes out of the one run: its messages, its answer and its token usage.
    tool_run_stats = extract_tool_run_stats(messages=result["messages"])
    answer = result["last_message"].text or ""
    usage = result.get("token_usage") or {}

    # Recall on the retrieved documents: how many of the expected documents the Agent found.
    retrieved_docs = result.get("documents") or []
    retrieved_ids = {document.id for document in retrieved_docs}
    found = case.expected_document_ids & retrieved_ids
    recall = len(found) / len(case.expected_document_ids)

    # Resolve each [doc <short-id>] the answer uses against what the Agent found. A reference matching nothing
    # is a fake citation, and one the answer doesn't use is an uncited document. Both are failures.
    cited_refs = _CITATION_RE.findall(answer)
    resolved = [ref for ref in cited_refs if any(document.id.startswith(ref) for document in retrieved_docs)]
    cited_ids = {document.id for document in retrieved_docs if any(document.id.startswith(r) for r in cited_refs)}
    citations_ok = len(resolved) == len(cited_refs)
    uncited = case.expected_document_ids - cited_ids

    # An eval case passes only on all four: it found every expected document, cited every one of them, made no
    # citation that does not resolve, and stayed inside its tool budget.
    spent = {tools: tool_run_stats.calls_to(tools=tools) for tools in case.tool_budgets}
    within_budget = all(used <= case.tool_budgets[tools] for tools, used in spent.items())
    passed = recall == 1.0 and not uncited and citations_ok and within_budget

    inspected_first = tool_run_stats.called_before(tools="list_metadata_fields", other=RETRIEVAL_TOOLS)
    counts = Counter(name for name, _ in tool_run_stats.calls)
    filters_used = [
        args["filters"] for name, args in tool_run_stats.calls if name in RETRIEVAL_TOOLS and args.get("filters")
    ]
    needed = len(case.expected_document_ids)
    print(f"\n=== eval case {position}/{total}: {'PASS' if passed else 'FAIL'} ===")
    print(f"  question: {case.question}")
    print(f"  tools called: {dict(counts)}")
    print(
        f"  inspected metadata first: {inspected_first}   retrievals with a filter: {len(filters_used)}   "
        f"tool errors: {len(tool_run_stats.errors)}   steps: {result['step_count']}   time: {elapsed:.1f}s"
    )
    for tools, used in spent.items():
        limit = case.tool_budgets[tools]
        label = tools if isinstance(tools, str) else " + ".join(tools)
        print(f"  tool budget: {label} {used}/{limit} -> {'ok' if used <= limit else 'EXCEEDED'}")
    print(f"  retrieval: found {len(found)}/{needed} of the documents the answer needs, {len(retrieved_docs)} returned")
    print(
        f"  citations: cited {needed - len(uncited)}/{needed} of them, "
        f"and {len(resolved)}/{len(cited_refs)} of the answer's references point at a returned document"
    )

    print("  documents returned, and whether the answer needs them:")
    for document in retrieved_docs:
        label = "needed" if document.id in case.expected_document_ids else "not needed"
        print(f"    {label:>10}  [doc {document.id[:8]}] {document.meta.get('title')}")
    # Naming the quote, since the id alone says nothing about what the run failed to find or failed to use.
    for document_id in sorted(case.expected_document_ids - retrieved_ids):
        print(f"  needed but never retrieved: [doc {document_id[:8]}] {case.evidence[document_id][:96]}")
    for document_id in sorted(uncited & retrieved_ids):
        print(f"  needed and retrieved but not cited: [doc {document_id[:8]}] {case.evidence[document_id][:96]}")

    if usage:
        print(f"  tokens: { {k: v for k, v in usage.items() if isinstance(v, int)} }")
    for tool_name, message in tool_run_stats.errors:
        print(f"  tool error: {tool_name} -> {message[:120]}")
    for filters in filters_used:
        print(f"  retrieval filter: {filters}")
    print("  answer:")
    for line in answer.splitlines():
        print(f"    {line}")
    return {"passed": passed, "usage": usage, "time": elapsed}


def main() -> None:
    """Run the eval set and print per-eval-case reports plus a summary."""
    parser = argparse.ArgumentParser(description="Mini evaluation harness for the Advanced RAG agent.")
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--max-cases", type=int, default=10, help="How many labelled questions to evaluate.")
    parser.add_argument("--case-seed", type=int, default=0, help="Selects which eval cases are drawn from the dataset.")
    arguments = parser.parse_args()

    store, articles = prepare_corpus(backend=arguments.store)
    chunks = sum(len(article.chunks) for article in articles.values())
    print(f"{CORPUS_KEY} on {arguments.store}: {chunks} chunks from {len(articles)} articles")

    labelled: list[LabelledQuestion] = build_eval_cases(
        articles=articles, limit=arguments.max_cases, seed=arguments.case_seed
    )
    # Budgets for the tools this agent has. Lenient on purpose: too many retrievals is better than too few.
    budgets: dict[str | tuple[str, ...], int] = {METADATA_TOOLS: 5, RETRIEVAL_TOOLS: 5}
    eval_cases = [
        EvalCase(question=question.question, evidence=question.evidence, tool_budgets=budgets) for question in labelled
    ]
    expected = sum(len(case.expected_document_ids) for case in eval_cases)
    print(f"eval cases: {len(eval_cases)} labelled from evidence, expecting {expected} documents in total")

    # Build the advanced rag agent
    agent = create_advanced_rag_agent(document_store=store, retriever=build_bm25_retriever(store=store))

    # Run the eval cases
    results = [
        run_eval_case(agent=agent, case=case, position=position, total=len(eval_cases))
        for position, case in enumerate(eval_cases, start=1)
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
