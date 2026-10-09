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

from haystack import Document
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from multihop_rag import CORPUS_KEY, LabelledQuestion, build_eval_cases, prepare_corpus
from util import build_bm25_retriever, preview

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent

RETRIEVAL_TOOLS = ("search_documents", "fetch_documents_by_filter")
METADATA_TOOLS = ("list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range")
_CITATION_RE = re.compile(r"\[doc ([0-9a-f]{4,16})[^]]*\]")


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
    filtered_retrievals = sum(
        1 for name, args in tool_run_stats.calls if name in RETRIEVAL_TOOLS and args.get("filters")
    )
    needed = len(case.expected_document_ids)
    print(f"\n=== eval case {position}/{total}: {'PASS' if passed else 'FAIL'} ===")
    print(f"  question: {case.question}")
    print(f"  tools called: {dict(counts)}")
    print(
        f"  inspected metadata first: {inspected_first}   retrievals with a filter: {filtered_retrievals}   "
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

    # Chunks of one article all carry its title, so they are grouped under it and told apart by chunk number.
    # Articles keep the order they were first returned in, which is the order the run ranked them.
    by_article: dict[str, list[Document]] = {}
    for document in retrieved_docs:
        by_article.setdefault(document.meta.get("title", ""), []).append(document)

    print("\n  documents returned, grouped by article. -> marks one the answer needs:")
    for title, documents in by_article.items():
        print(f"    Title: {title}")
        for document in sorted(documents, key=lambda chunk: chunk.meta.get("split_id", 0)):
            needed_here = "-> " if document.id in case.expected_document_ids else "   "
            snippet = preview(text=document.content or "", limit=80)
            print(f"      {needed_here}chunk {document.meta.get('split_id'):>2}  [doc {document.id[:8]}]  {snippet}")
    # Naming the quote, since the id alone says nothing about what the run failed to find or failed to use.
    print()
    for document_id in sorted(case.expected_document_ids - retrieved_ids):
        print(
            f"  needed but never retrieved: [doc {document_id[:8]}] "
            f"{preview(text=case.evidence[document_id], limit=96)}"
        )
    for document_id in sorted(uncited & retrieved_ids):
        print(
            f"  needed and retrieved but not cited: [doc {document_id[:8]}] "
            f"{preview(text=case.evidence[document_id], limit=96)}"
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
    # Generous on purpose: a MultiHopRAG question needs evidence from several articles, so several searches are
    # the expected shape of a good run rather than a sign of floundering.
    budgets: dict[str | tuple[str, ...], int] = {METADATA_TOOLS: 8, RETRIEVAL_TOOLS: 12}
    eval_cases = [
        EvalCase(question=question.question, evidence=question.evidence, tool_budgets=budgets) for question in labelled
    ]
    print(f"eval cases: {len(eval_cases)} labelled from evidence")

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
