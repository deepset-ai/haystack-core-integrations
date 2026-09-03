# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Run an Agent-configuration optimization experiment against a realistic large RAG corpus.

The PoC uses the same roughly 150,000-document Amazon Reviews 2023 corpus and metadata-constrained cases as
`advanced_rag_eval.py`. It records successful reference runs, lets an optimizer Agent edit the complete serialized
candidate configuration, evaluates each choice, and feeds the measured outcome into the next choice. Nothing is
deployed automatically.

Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:

    hatch run test:python examples/harness_optimization_poc.py
    hatch run test:python examples/harness_optimization_poc.py --max-cases 1 --max-iterations 1
    hatch run test:python examples/harness_optimization_poc.py --store opensearch
    hatch run test:python examples/harness_optimization_poc.py --docs-mcp

The default evaluates two cases and at most three candidates. A persistent OpenSearch store avoids rebuilding the
corpus between invocations. Journaled measurements are reused until the Agent, corpus identity, recorded runs, or
evaluator configuration changes.
"""

import argparse
import os
import shutil
from pathlib import Path
from uuid import uuid4

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DocumentStore
from haystack.tools import flatten_tools_or_toolsets
from util import (
    LARGE_CASES,
    LARGE_CORPUS_CATEGORIES,
    LARGE_CORPUS_DOCS_PER_CATEGORY,
    build_document_store,
    build_retriever,
    populate_corpus,
)

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
    question_from_messages,
)
from haystack_integrations.agent_pack.dataclasses import AgentRunRecord
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    ExperimentResult,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)

WORKSPACE = Path(".agent-pack-poc")
REFERENCE_MODEL = "gpt-5.6-sol"
CANDIDATE_MODELS = ("gpt-5.6-terra", "gpt-5.6-luna")

# USD prices per million tokens, used only to rank candidates against each other.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}


def build_reference_agent(store: DocumentStore, model: str) -> Agent:
    """Build the Advanced RAG Agent whose complete configuration will be optimized."""
    return create_advanced_rag_agent(
        document_store=store,
        retriever=build_retriever(store=store),
        llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
        # Keep backup-answer usage attributable to the selected reference model. Changing only the coordinator's
        # model path leaves this fallback unchanged unless the optimizer explicitly edits it too.
        backup_answer_llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
    )


def capture_reference_runs(
    agent: Agent, cases: list[AdvancedRAGEvaluationCase], run_store: LocalRunStore
) -> frozenset[str]:
    """Run and persist one successful reference input/output pair for every selected case."""
    records_by_question = {
        question_from_messages(messages=record.inputs.get("messages") or []): record for record in run_store.list()
    }
    selected_ids: set[str] = set()
    for case in cases:
        if existing := records_by_question.get(case.question):
            print(f"  reusing: {case.question}")
            selected_ids.add(existing.run_id)
            continue
        print(f"  capturing: {case.question}")
        messages = [ChatMessage.from_user(text=case.question)]
        result = agent.run(messages=messages)
        record = AgentRunRecord(run_id=str(uuid4()), inputs={"messages": messages}, outputs=result)
        run_store.add(record=record)
        selected_ids.add(record.run_id)
        answer = result["last_message"].text or ""
        print(f"    run={record.run_id[:8]} answer={answer[:90]!r}")
    return frozenset(selected_ids)


def build_pricing(models: tuple[str, ...]) -> ModelPriceCatalog:
    """Build informational prices for the models currently known to this example."""
    return ModelPriceCatalog(
        prices=[
            ModelPrice(
                model_id=model,
                input_cost_per_million=MODEL_PRICES[model][0],
                output_cost_per_million=MODEL_PRICES[model][1],
            )
            for model in models
            if model in MODEL_PRICES
        ],
    )


def format_cost(cost: float | None) -> str:
    """Format a measured cost that may be unavailable for an optimizer-selected model."""
    return "unpriced" if cost is None else f"${cost:.6f}"


def report(result: ExperimentResult) -> None:
    """Print baseline, candidate, gate, and recommendation details."""
    baseline = result.baseline
    print("\n--- baseline (reference Agent) ---")
    print(
        f"  quality={baseline.quality:.2f} cost={format_cost(cost=baseline.cost)} "
        f"latency={baseline.latency_ms:.0f}ms model={baseline.details.get('model')}"
    )
    for case_metrics in baseline.details.get("cases", []):
        verdict = "pass" if case_metrics["passed"] else "FAIL " + ",".join(case_metrics["failures"])
        print(f"    [{verdict}] {case_metrics['question']}")

    print("\n--- candidates ---")
    for candidate in result.candidates:
        gates = result.gate_failures.get(candidate.candidate_id, ())
        if candidate.metrics is None:
            print(f"  {candidate.mutation} -> failed: {candidate.failure}")
            continue
        print(
            f"  {candidate.mutation} -> quality={candidate.metrics.quality:.2f} "
            f"cost={format_cost(cost=candidate.metrics.cost)} latency={candidate.metrics.latency_ms:.0f}ms"
        )
        print(f"    gates: {'passed' if not gates else ', '.join(gates)}")
        for case_metrics in candidate.metrics.details.get("cases", []):
            if not case_metrics["passed"]:
                print(f"      regression on {case_metrics['question']!r}: {','.join(case_metrics['failures'])}")

    print("\n--- recommendation ---")
    if result.recommendation is None:
        print("  none: no candidate cleared every gate and improved on the reference")
        return
    recommendation = result.recommendation
    print(f"  mutation: {recommendation.evaluation.mutation}")
    print(f"  reasons: {', '.join(recommendation.reasons)}")
    recommendation_metrics = recommendation.evaluation.metrics
    if baseline.cost is not None and recommendation_metrics is not None and recommendation_metrics.cost is not None:
        print(f"  cost saving on this evaluation set: ${baseline.cost - recommendation_metrics.cost:.6f}")
    print("  Nothing was deployed. Approving this recommendation is a separate, human decision.")


def parse_args() -> argparse.Namespace:
    """Parse the PoC command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--reference-model", default=REFERENCE_MODEL)
    parser.add_argument("--candidate-model", action="append", dest="candidate_models")
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument(
        "--documents-per-category",
        type=int,
        default=LARGE_CORPUS_DOCS_PER_CATEGORY,
        help="Reviews streamed for each of the three categories. Lower this only for smoke testing.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Times to evaluate each case per configuration. More repetitions reduce sensitivity to variable runs.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=1.0,
        help="Minimum candidate case pass rate in the inclusive range 0.0 to 1.0.",
    )
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.0,
        help="Maximum absolute pass-rate loss from the reference, between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--primary",
        choices=("cost", "latency"),
        default="cost",
        help="Primary measurement to minimize after candidate quality gates pass.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of candidate configurations included in the experiment.",
    )
    parser.add_argument(
        "--docs-mcp",
        action="store_true",
        help="Give the optimizer Agent access to the public Haystack documentation MCP server.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove the saved runs and experiment journal before starting.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the large-corpus experiment and run it end to end."""
    arguments = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY must be set to run this walkthrough."
        raise SystemExit(message)
    if arguments.max_cases < 1:
        message = "--max-cases must be at least 1."
        raise SystemExit(message)
    if arguments.documents_per_category < 1:
        message = "--documents-per-category must be at least 1."
        raise SystemExit(message)
    if arguments.fresh and WORKSPACE.exists():
        shutil.rmtree(path=WORKSPACE)

    print("=== 1. set up large corpus ===")
    corpus_key = (
        "large"
        if arguments.documents_per_category == LARGE_CORPUS_DOCS_PER_CATEGORY
        else f"large-{arguments.documents_per_category}"
    )
    store = build_document_store(backend=arguments.store, corpus=corpus_key)
    if store.count_documents() == 0:
        populate_corpus(
            store=store,
            corpus="large",
            documents_per_category=arguments.documents_per_category,
        )
    else:
        print("  store already populated, skipping indexing")
    document_count = store.count_documents()
    print(f"  amazon-reviews-2023 on {arguments.store}: {document_count} documents")

    selected_definitions = LARGE_CASES[: arguments.max_cases]
    cases = [definition.to_optimization_case() for definition in selected_definitions]
    candidate_models = tuple(arguments.candidate_models or CANDIDATE_MODELS)
    reference_agent = build_reference_agent(store=store, model=arguments.reference_model)
    tool_names = sorted(configured.name for configured in flatten_tools_or_toolsets(tools=reference_agent.tools))
    print(f"  model={arguments.reference_model} tools={tool_names}")

    print("\n=== 2. execute and store reference runs ===")
    run_store = LocalRunStore(directory=WORKSPACE / "runs")
    selected_run_ids = capture_reference_runs(agent=reference_agent, cases=cases, run_store=run_store)

    pricing = build_pricing(models=(arguments.reference_model, *candidate_models))

    print("\n=== 3. optimization experiment ===")
    docs_toolset = create_haystack_documentation_mcp_toolset() if arguments.docs_mcp else None
    experiment = HarnessOptimizationExperiment(
        reference=reference_agent,
        run_store=run_store,
        evaluator=AdvancedRAGHarnessEvaluator(cases=cases, repetitions=arguments.repetitions),
        pricing=pricing,
        objectives=OptimizationObjectives(
            min_quality=arguments.min_quality,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=ExperimentJournal(path=WORKSPACE / "experiment.jsonl"),
        optimizer_agent=create_harness_optimizer_agent(docs_toolset=docs_toolset),
        run_ids=selected_run_ids,
        max_iterations=arguments.max_iterations,
        configuration_key=(
            f"amazon-reviews-2023:{','.join(LARGE_CORPUS_CATEGORIES)}:"
            f"{arguments.documents_per_category}:{document_count}"
        ),
    )
    result = experiment.run()
    print(f"  configuration hash: {result.configuration_hash[:16]}")

    print("\n=== 5. outcome ===")
    report(result=result)
    print(f"\nJournal: {WORKSPACE / 'experiment.jsonl'} (re-running resumes measured candidates)")


if __name__ == "__main__":
    main()
