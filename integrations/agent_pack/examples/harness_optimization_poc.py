# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end walkthrough of Agent Pack's experimental harness optimization API.

The script does, against real models, everything the feature claims:

1. Builds an Advanced RAG agent over a small in-memory corpus. This is the champion harness every candidate is
   measured against, and it is never mutated.
2. Captures a successful run per evaluation question with `TraceCapturingAgentRunner`, writing `haystack-trace/v1`
   artifacts to disk. Content capture stays local: an already-installed tracer keeps exporting exactly what it did
   before.
3. Declares an approved asset catalog — which models and tools a candidate may use, and what they cost. The catalog
   is the control: only what you list here can end up in a candidate.
4. Runs a `HarnessOptimizationCampaign`: it measures the reference, materializes one candidate per approved
   alternative model through `Agent.clone`, validates each candidate's assets before it executes, replays the
   captured questions, and ranks whatever clears the quality gate.
5. Prints whether the champion itself is catalog-compliant, the baseline, every candidate's measurements, the gates
   each one missed, and the recommendation with the reasons behind it. Nothing is promoted: the recommendation is
   materialized only so you can inspect it.

Run it from the integration directory (`integrations/agent_pack`) with `OPENAI_API_KEY` set:

    hatch run test:python examples/harness_optimization_poc.py
    hatch run test:python examples/harness_optimization_poc.py --max-cases 1
    hatch run test:python examples/harness_optimization_poc.py --repetitions 3
    hatch run test:python examples/harness_optimization_poc.py --proposer agent --docs-mcp
    hatch run test:python examples/harness_optimization_poc.py --drop-approved-tool get_metadata_field_range

Cost: one reference measurement plus one per candidate model, each replaying every question `--repetitions` times.
With the defaults that is 3 questions x 2 models = 6 agent runs. The campaign journal makes a re-run resume, so an
interrupted run does not pay for completed candidates twice.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from haystack import Document
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.evaluation import AdvancedRAGEvaluationCase
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import AdvancedRAGHarnessEvaluator
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    CampaignJournal,
    CampaignResult,
    HarnessOptimizationCampaign,
    HarnessOptimizerAgentProposer,
    ModelAsset,
    OptimizationObjectives,
    ToolAsset,
    create_harness_optimizer_agent,
    create_haystack_docs_toolset,
)
from haystack_integrations.agent_pack.optimization.recipes import RECIPE_PROPOSAL_JSON_SCHEMA
from haystack_integrations.agent_pack.tracing import (
    LocalTraceCollector,
    LocalTraceStore,
    TraceCapturingAgentRunner,
)

WORKSPACE = Path(".agent-pack-poc")

#: The reference harness runs the stronger model; every other entry is a candidate the campaign will try.
REFERENCE_MODEL = "gpt-5"
CANDIDATE_MODELS = ("gpt-5-mini",)

#: Illustrative USD prices per million tokens, only used to rank candidates against each other. Replace them with
#: your own contracted rates before reading anything into the absolute numbers.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
}


def build_corpus() -> InMemoryDocumentStore:
    """
    Create the small benchmark corpus, with the varied metadata the Advanced RAG tools inspect.

    :returns: A populated in-memory document store.
    """
    documents = [
        Document(
            content="CRISPR gene editing corrected hereditary blindness mutations in a 2021 clinical trial.",
            meta={"category": "science", "year": 2021, "language": "en"},
        ),
        Document(
            content="A quantum processor achieved error-corrected logical qubits in 2023.",
            meta={"category": "science", "year": 2023, "language": "en"},
        ),
        Document(
            content="Alloy design for fusion reactor walls advanced sharply in 2019.",
            meta={"category": "science", "year": 2019, "language": "en"},
        ),
        Document(
            content="The Berlin Wall fell in 1989, reuniting the city.",
            meta={"category": "history", "year": 1989, "language": "en"},
        ),
        Document(
            content="Apollo 11 landed the first humans on the Moon in 1969.",
            meta={"category": "history", "year": 1969, "language": "en"},
        ),
        Document(
            content="Die Wiedervereinigung Deutschlands wurde 1990 vollzogen.",
            meta={"category": "history", "year": 1990, "language": "de"},
        ),
    ]
    store = InMemoryDocumentStore()
    store.write_documents(documents)
    return store


def build_cases(store: InMemoryDocumentStore) -> list[AdvancedRAGEvaluationCase]:
    """
    Label the evaluation set against the corpus.

    Expected documents are resolved from metadata here rather than hand-copied, so the labels stay correct if the
    corpus changes. Labelled cases are what make a recommendation trustworthy: without them the campaign can only
    check that a candidate retrieves the same documents the incumbent did, which measures imitation, not quality.

    :param store: The populated corpus.
    :returns: One case per question.
    """
    documents = store.filter_documents()

    def ids_where(**constraints: Any) -> frozenset[str]:
        return frozenset(
            document.id
            for document in documents
            if all(document.meta.get(key) == value for key, value in constraints.items())
        )

    science_after_2015 = frozenset(
        document.id
        for document in documents
        if document.meta.get("category") == "science" and document.meta.get("year", 0) > 2015
    )
    history_before_1990 = frozenset(
        document.id
        for document in documents
        if document.meta.get("category") == "history" and document.meta.get("year", 9999) < 1990
    )

    return [
        AdvancedRAGEvaluationCase(
            question="What scientific breakthroughs happened after 2015, according to the documents?",
            expected_document_ids=science_after_2015,
            answer_must_mention=("CRISPR", "quantum"),
        ),
        AdvancedRAGEvaluationCase(
            question="Which historical events in the documents happened before 1990?",
            expected_document_ids=history_before_1990,
            answer_must_mention=("Berlin", "Apollo"),
        ),
        AdvancedRAGEvaluationCase(
            question="What does the German-language document describe?",
            expected_document_ids=ids_where(language="de"),
            answer_must_mention=("reunification",),
            # One document is enough here, so a broader sweep is a process regression worth catching.
            max_retrieval_calls=3,
        ),
    ]


def build_reference_agent(*, store: InMemoryDocumentStore, model: str) -> Agent:
    """
    Build the champion harness.

    :param store: The corpus to retrieve from.
    :param model: The reference model.
    :returns: The reference Agent.
    """
    return create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store, top_k=5),
        llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
        # Passed explicitly so the whole harness runs the model this script chose. Left to its default, the agent
        # builds its backup-answer hook on a second model, which the asset catalog then rightly rejects as
        # undeclared. Note that a model substitution replaces the coordinator generator only, so a candidate keeps
        # this backup model; it costs nothing unless a run is cut off by `max_agent_steps`.
        backup_answer_llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
    )


def capture_reference_runs(
    *, agent: Agent, cases: list[AdvancedRAGEvaluationCase], trace_store: LocalTraceStore
) -> None:
    """
    Run the reference harness once per question and persist the captured traces.

    :param agent: The reference Agent.
    :param cases: The questions to capture.
    :param trace_store: Where the artifacts are written.
    """
    runner = TraceCapturingAgentRunner(collector=LocalTraceCollector(store=trace_store, capture_content=True))
    for case in cases:
        print(f"  capturing: {case.question}")
        captured = runner.run(agent, messages=[ChatMessage.from_user(case.question)])
        answer = captured.result["last_message"].text or ""
        print(f"    status={captured.trace.status} spans={len(captured.trace.traces)} answer={answer[:90]!r}")


def build_catalog(
    *, agent: Agent, models: tuple[str, ...], dropped_tools: tuple[str, ...] = ()
) -> ApprovedAssetCatalog:
    """
    Declare which models and tools a candidate may use.

    The catalog is the only compliance control: a candidate configured with anything outside it is rejected while it
    is being materialized and never executes.

    :param agent: The reference Agent, read for the tools it already exposes.
    :param models: Every approved model, reference included.
    :param dropped_tools: Tools to leave out of the catalog, to show a candidate being rejected before it runs.
    :returns: The approved asset catalog.
    """
    return ApprovedAssetCatalog(
        models=[
            ModelAsset(
                model_id=model,
                provider="openai",
                deployment="hosted",
                input_cost_per_million=MODEL_PRICES.get(model, (0.0, 0.0))[0],
                output_cost_per_million=MODEL_PRICES.get(model, (0.0, 0.0))[1],
            )
            for model in models
        ],
        tools=[
            ToolAsset(name=configured.name)
            for configured in flatten_tools_or_toolsets(tools=agent.tools)
            if configured.name not in dropped_tools
        ],
    )


def report(*, result: CampaignResult, reference: Agent, assets: ApprovedAssetCatalog) -> None:
    """
    Print the campaign outcome.

    :param result: What the campaign measured.
    :param reference: The champion harness, shown to be unchanged.
    :param assets: The approved asset catalog, used to re-validate the recommendation.
    """
    validation = result.reference_validation
    if validation is not None and not validation.allowed:
        print("\n--- reference harness is NOT catalog-compliant ---")
        print(f"  {', '.join(validation.violations)}")
        print("  It is still measured, so you can see what replacing it would save.")

    baseline = result.baseline
    print("\n--- baseline (reference harness) ---")
    print(
        f"  quality={baseline.quality:.2f} cost=${baseline.cost:.6f} latency={baseline.latency_ms:.0f}ms "
        f"model={baseline.details.get('model')}"
    )
    for case_metrics in baseline.details.get("cases", []):
        verdict = "pass" if case_metrics["passed"] else "FAIL " + ",".join(case_metrics["failures"])
        print(f"    [{verdict}] {case_metrics['question']}")

    print("\n--- candidates ---")
    for candidate in result.candidates:
        gates = result.gate_failures.get(candidate.candidate_id, ())
        if candidate.metrics is None:
            print(f"  {candidate.recipe} -> failed: {candidate.failure}")
            continue
        print(
            f"  {candidate.recipe} -> quality={candidate.metrics.quality:.2f} "
            f"cost=${candidate.metrics.cost:.6f} latency={candidate.metrics.latency_ms:.0f}ms"
        )
        print(f"    assets: models={candidate.asset_validation.model_ids if candidate.asset_validation else ()}")
        print(f"    gates: {'passed' if not gates else ', '.join(gates)}")
        for case_metrics in candidate.metrics.details.get("cases", []):
            if not case_metrics["passed"]:
                print(f"      regression on {case_metrics['question']!r}: {','.join(case_metrics['failures'])}")

    print("\n--- recommendation ---")
    if result.recommendation is None:
        print("  none: no candidate cleared every gate and improved on the reference")
        return
    recommendation = result.recommendation
    print(f"  recipe: {recommendation.evaluation.recipe}")
    print(f"  reasons: {', '.join(recommendation.reasons)}")
    saving = baseline.cost - (recommendation.evaluation.metrics.cost if recommendation.evaluation.metrics else 0.0)
    print(f"  cost saving on this evaluation set: ${saving:.6f}")
    if "quality_unvalidated" in recommendation.reasons:
        print("  NOTE: quality was scored against cases derived from the reference traces, not labelled ones.")

    candidate = recommendation.materialize(reference, assets)
    print(f"  materialized candidate model: {candidate.chat_generator.model}")
    print(f"  reference model, unchanged:   {reference.chat_generator.model}")
    print("  Nothing was deployed. Approving this recommendation is a separate, human decision.")


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    :returns: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-model", default=REFERENCE_MODEL, help="Model the champion harness runs.")
    parser.add_argument(
        "--candidate-model",
        action="append",
        dest="candidate_models",
        help=f"Approved alternative model. Repeatable. Defaults to {', '.join(CANDIDATE_MODELS)}.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Evaluate only the first N questions. Use it for a cheap smoke run over the whole pipeline.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="How many times each question is replayed per harness. Above 1, the quality gate compares pessimistic "
        "lower bounds instead of single noisy samples.",
    )
    parser.add_argument("--min-quality", type=float, default=1.0, help="Absolute quality floor for a candidate.")
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.0,
        help="How far below the reference's quality a candidate may fall.",
    )
    parser.add_argument(
        "--primary", choices=("cost", "latency"), default="cost", help="What candidates are ranked on first."
    )
    parser.add_argument(
        "--proposer",
        choices=("models", "agent"),
        default="models",
        help="'models' enumerates every approved alternative model. 'agent' asks a skill-guided optimizer Agent for "
        "typed recipes, which are still rejected unless they parse as supported transformations.",
    )
    parser.add_argument(
        "--docs-mcp",
        action="store_true",
        help="Give the optimizer Agent read-only access to the public Haystack documentation MCP server. Requires "
        "mcp-haystack.",
    )
    parser.add_argument(
        "--drop-approved-tool",
        action="append",
        dest="dropped_tools",
        default=[],
        help="Tool to leave out of the approved catalog, so every candidate is rejected before it runs. Repeatable.",
    )
    parser.add_argument("--fresh", action="store_true", help="Delete captured traces and the journal first.")
    return parser.parse_args()


def main() -> None:
    """Run the walkthrough."""
    arguments = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY must be set to run this walkthrough."
        raise SystemExit(message)

    if arguments.fresh and WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    candidate_models = tuple(arguments.candidate_models or CANDIDATE_MODELS)
    dropped_tools = tuple(arguments.dropped_tools)

    print("=== 1. reference harness ===")
    store = build_corpus()
    cases = build_cases(store)[: arguments.max_cases]
    reference = build_reference_agent(store=store, model=arguments.reference_model)
    tool_names = sorted(configured.name for configured in flatten_tools_or_toolsets(tools=reference.tools))
    print(f"  model={arguments.reference_model} tools={tool_names}")

    print("\n=== 2. capture successful reference runs ===")
    trace_store = LocalTraceStore(directory=WORKSPACE / "traces")
    if len(trace_store.list()) < len(cases):
        capture_reference_runs(agent=reference, cases=cases, trace_store=trace_store)
    else:
        print(f"  reusing {len(trace_store.list())} captured traces from {WORKSPACE / 'traces'}")

    print("\n=== 3. approved assets ===")
    assets = build_catalog(
        agent=reference, models=(arguments.reference_model, *candidate_models), dropped_tools=dropped_tools
    )
    if dropped_tools:
        print(f"  deliberately left out of the catalog: {list(dropped_tools)}")
    for asset in assets.models.values():
        print(
            f"  model {asset.model_id} ({asset.provider}/{asset.deployment}): "
            f"in=${asset.input_cost_per_million}/M out=${asset.output_cost_per_million}/M"
        )

    print("\n=== 4. campaign ===")
    proposer = None
    if arguments.proposer == "agent":
        docs_toolset = create_haystack_docs_toolset() if arguments.docs_mcp else None
        proposer = HarnessOptimizerAgentProposer(
            optimizer_agent=create_harness_optimizer_agent(
                # Structured output keeps the proposal well-formed; every recipe is still validated on the way in.
                chat_generator=OpenAIResponsesChatGenerator(
                    model=arguments.reference_model,
                    generation_kwargs={
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "harness_optimizer_proposal",
                                "schema": RECIPE_PROPOSAL_JSON_SCHEMA,
                                "strict": False,
                            }
                        }
                    },
                ),
                docs_toolset=docs_toolset,
            ),
            max_recipes=4,
        )
        print("  proposer: skill-guided optimizer Agent (proposals are parsed, never executed as code)")
    else:
        print("  proposer: deterministic enumeration of approved models")

    campaign = HarnessOptimizationCampaign(
        reference=reference,
        trace_source=trace_store,
        evaluator=AdvancedRAGHarnessEvaluator(cases=cases, repetitions=arguments.repetitions),
        assets=assets,
        objectives=OptimizationObjectives(
            min_quality=arguments.min_quality,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=CampaignJournal(path=WORKSPACE / "campaign.jsonl"),
        proposer=proposer,
        # The corpus is not visible in the harness configuration, so it is named explicitly: change the corpus and
        # journaled measurements are invalidated instead of silently reused.
        configuration_key="poc-corpus-v1",
    )
    result = campaign.run()
    print(f"  configuration hash: {result.configuration_hash[:16]}")

    print("\n=== 5. outcome ===")
    report(result=result, reference=reference, assets=assets)
    print(f"\nJournal: {WORKSPACE / 'campaign.jsonl'} (re-running resumes completed candidates)")


if __name__ == "__main__":
    main()
