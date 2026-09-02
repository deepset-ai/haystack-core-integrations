# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runs an optimization experiment end to end against real models.

The script:

1. Builds an Advanced RAG agent over a small in-memory corpus. This is the agent being optimized: every candidate
   is a variant of it, and it is never modified.
2. Records one successful input/output pair per evaluation question, writing compact run records to disk.
3. Declares known model prices. They help rank measured candidates but do not restrict what the optimizer may edit.
4. Runs a `HarnessOptimizationExperiment`, whose optimizer Agent chooses one candidate, observes its measurement,
   and uses that evidence to choose the next until it stops or reaches the iteration budget.
5. Prints the baseline, each candidate's measurements, the gates it missed, and the recommendation. Nothing is
   promoted; the recommendation is materialized only so you can inspect it.

Run it from the integration directory (`integrations/agent_pack`) with `OPENAI_API_KEY` set:

    hatch run test:python examples/harness_optimization_poc.py
    hatch run test:python examples/harness_optimization_poc.py --max-cases 1
    hatch run test:python examples/harness_optimization_poc.py --repetitions 3
    hatch run test:python examples/harness_optimization_poc.py --docs-mcp

Cost: one reference measurement plus up to `--max-iterations` candidate measurements, each replaying every question
`--repetitions` times. The optimizer may stop earlier. Results are journaled, so a re-run skips completed work until
the harness, recorded runs, evaluator, or experiment configuration changes.
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
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
    question_from_messages,
)
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    ExperimentResult,
    HarnessOptimizationExperiment,
    HarnessOptimizerAgentProposer,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.runs import AgentRunRecorder, LocalRunStore, RunSelection

WORKSPACE = Path(".agent-pack-poc")

#: The agent being optimized runs the most capable tier; the others are candidates the experiment will try. The
#: GPT-5.6 family is named rather than numbered: Sol is the most capable, Terra sits in the middle, and Luna is the
#: cheapest and fastest. Two candidates are offered so the ranking has something to choose between. Pass
#: `--reference-model` and `--candidate-model` to try others.
REFERENCE_MODEL = "gpt-5.6-sol"
CANDIDATE_MODELS = ("gpt-5.6-terra", "gpt-5.6-luna")

#: USD prices per million tokens, used only to rank candidates against each other. These are OpenAI's published
#: list prices; replace them with your own contracted rates before reading anything into the absolute numbers.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
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
    corpus changes. Labelled cases are what make a recommendation trustworthy: without them the experiment can only
    check that a candidate retrieves the same documents the incumbent did, which measures imitation, not quality.

    :param store: The populated corpus.
    :returns: One case per question.
    """
    documents = store.filter_documents()

    def ids_where(**constraints: Any) -> frozenset[str]:
        """Return IDs of corpus documents whose metadata matches every constraint."""
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


def build_reference_agent(store: InMemoryDocumentStore, model: str) -> Agent:
    """
    Build the agent to be optimized.

    :param store: The corpus to retrieve from.
    :param model: The reference model.
    :returns: The reference Agent.
    """
    return create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store, top_k=5),
        llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
        # Passed explicitly so the whole harness runs the model this script chose. Left to its default, the agent
        # builds its backup-answer hook on a second model, which would then be priced as an unknown model. Note
        # that changing the coordinator's model path alone keeps this backup model; it costs nothing unless a run is
        # cut off by `max_agent_steps`.
        backup_answer_llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs={"reasoning": {"effort": "low"}}),
    )


def capture_reference_runs(
    agent: Agent, cases: list[AdvancedRAGEvaluationCase], run_store: LocalRunStore
) -> RunSelection:
    """
    Run the reference harness once per question and persist its inputs and outputs.

    :param agent: The reference Agent.
    :param cases: The questions to capture.
    :param run_store: Where input/output records are written.
    :returns: A selection containing exactly one run for each requested question.
    """
    records_by_question = {
        question_from_messages(messages=record.inputs.get("messages") or []): record for record in run_store.list()
    }
    recorder = AgentRunRecorder(store=run_store)
    selected_ids: set[str] = set()
    for case in cases:
        if existing := records_by_question.get(case.question):
            print(f"  reusing: {case.question}")
            selected_ids.add(existing.run_id)
            continue
        print(f"  capturing: {case.question}")
        recorded = recorder.run(agent=agent, messages=[ChatMessage.from_user(text=case.question)])
        selected_ids.add(recorded.record.run_id)
        answer = recorded.result["last_message"].text or ""
        print(f"    run={recorded.record.run_id[:8]} answer={answer[:90]!r}")
    return RunSelection(run_ids=frozenset(selected_ids))


def build_pricing(models: tuple[str, ...]) -> ModelPriceCatalog:
    """
    Declare currently known model prices without limiting optimizer choices.

    :param models: Models whose measured token usage can currently be priced.
    :returns: Informational pricing for the experiment.
    """
    return ModelPriceCatalog(
        prices=[
            ModelPrice(
                model_id=model,
                input_cost_per_million=MODEL_PRICES.get(model, (0.0, 0.0))[0],
                output_cost_per_million=MODEL_PRICES.get(model, (0.0, 0.0))[1],
            )
            for model in models
            if model in MODEL_PRICES
        ],
    )


def format_cost(cost: float | None) -> str:
    """
    Format a measured cost that may be unavailable for an optimizer-selected model.

    :param cost: The priced cost, or ``None`` when model pricing is unknown.
    :returns: A display-safe cost.
    """
    return "unpriced" if cost is None else f"${cost:.6f}"


def report(result: ExperimentResult, reference: Agent) -> None:
    """
    Print the experiment outcome.

    :param result: What the experiment measured.
    :param reference: The Agent being optimized, printed to show it was left unchanged.
    """
    baseline = result.baseline
    print("\n--- baseline (reference harness) ---")
    print(
        f"  quality={baseline.quality:.2f} cost={format_cost(cost=baseline.cost)} "
        f"latency={baseline.latency_ms:.0f}ms "
        f"model={baseline.details.get('model')}"
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
        saving = baseline.cost - recommendation_metrics.cost
        print(f"  cost saving on this evaluation set: ${saving:.6f}")
    if "quality_unvalidated" in recommendation.reasons:
        print("  NOTE: quality was scored against cases derived from reference runs, not labelled ones.")

    candidate = recommendation.materialize(reference=reference)
    print(f"  materialized candidate model: {candidate.chat_generator.model}")
    print(f"  reference model, unchanged:   {reference.chat_generator.model}")
    print("  Nothing was deployed. Approving this recommendation is a separate, human decision.")


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    :returns: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-model", default=REFERENCE_MODEL, help="Model the agent being optimized runs.")
    parser.add_argument(
        "--candidate-model",
        action="append",
        dest="candidate_models",
        help=f"Alternative model with known pricing. Repeatable. Defaults to {', '.join(CANDIDATE_MODELS)}.",
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
    parser.add_argument("--max-iterations", type=int, default=6, help="Maximum candidates the optimizer may measure.")
    parser.add_argument(
        "--docs-mcp",
        action="store_true",
        help="Give the optimizer Agent read-only access to the Haystack documentation MCP server. Requires "
        "mcp-haystack.",
    )
    parser.add_argument("--fresh", action="store_true", help="Delete recorded runs and the journal first.")
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

    print("=== 1. reference harness ===")
    store = build_corpus()
    cases = build_cases(store=store)[: arguments.max_cases]
    reference = build_reference_agent(store=store, model=arguments.reference_model)
    tool_names = sorted(configured.name for configured in flatten_tools_or_toolsets(tools=reference.tools))
    print(f"  model={arguments.reference_model} tools={tool_names}")

    print("\n=== 2. capture successful reference runs ===")
    run_store = LocalRunStore(directory=WORKSPACE / "runs")
    run_selection = capture_reference_runs(agent=reference, cases=cases, run_store=run_store)

    print("\n=== 3. known model prices (informational) ===")
    pricing = build_pricing(models=(arguments.reference_model, *candidate_models))
    for price in pricing.prices.values():
        print(f"  model {price.model_id}: in=${price.input_cost_per_million}/M out=${price.output_cost_per_million}/M")

    print("\n=== 4. experiment ===")
    docs_toolset = create_haystack_documentation_mcp_toolset() if arguments.docs_mcp else None
    proposer = HarnessOptimizerAgentProposer(
        # The generator defaults to the pack's own choice, so the model being measured and the model doing the
        # proposing stay independent.
        optimizer_agent=create_harness_optimizer_agent(docs_toolset=docs_toolset),
    )
    print("  proposer: iterative optimizer Agent with baseline and prior measurements")

    experiment = HarnessOptimizationExperiment(
        reference=reference,
        run_source=run_store,
        evaluator=AdvancedRAGHarnessEvaluator(cases=cases, repetitions=arguments.repetitions),
        pricing=pricing,
        objectives=OptimizationObjectives(
            min_quality=arguments.min_quality,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=ExperimentJournal(path=WORKSPACE / "experiment.jsonl"),
        proposer=proposer,
        run_selection=run_selection,
        max_iterations=arguments.max_iterations,
        # The corpus is not visible in the harness configuration, so it is named explicitly: change the corpus and
        # journaled measurements are invalidated instead of silently reused.
        configuration_key="poc-corpus-v1",
    )
    result = experiment.run()
    print(f"  configuration hash: {result.configuration_hash[:16]}")

    print("\n=== 5. outcome ===")
    report(result=result, reference=reference)
    print(f"\nJournal: {WORKSPACE / 'experiment.jsonl'} (re-running resumes measured candidates)")


if __name__ == "__main__":
    main()
