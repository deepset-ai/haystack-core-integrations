# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Run an Agent-configuration optimization experiment against a labelled RAG evaluation set.
#
# The corpus and eval cases come from `multihop_rag`, which chunks the MultiHopRAG news articles and derives
# each eval case's expected documents from which chunks contain its labelled evidence. The run lets an
# optimizer Agent edit the complete serialized candidate configuration, evaluates each choice against those
# eval cases, and feeds the measured outcome into the next choice. Nothing is deployed automatically.
#
# The reference Agent is deliberately badly configured, so the run shows whether the optimizer can build a better one
# from measured evidence. See `POOR_RETRIEVER_TOP_K` and the constants next to it for what is wrong with it and why.
#
# Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:
#
#     hatch run test:python examples/advanced_rag_harness_optimization.py
#     hatch run test:python examples/advanced_rag_harness_optimization.py --max-eval-cases 1 --max-iterations 1
#     hatch run test:python examples/advanced_rag_harness_optimization.py --primary quality --max-quality-loss 0.05
#     hatch run test:python examples/advanced_rag_harness_optimization.py --store opensearch
#     hatch run test:python examples/advanced_rag_harness_optimization.py --docs-mcp
#
# Every eval case is evaluated once per candidate, so a candidate costs `--max-eval-cases` Agent runs and quality is the
# fraction of eval cases it passed. A persistent OpenSearch store avoids rebuilding the corpus between invocations.
# Each invocation measures its own reference and its own candidates, and records them to the journal; nothing is
# carried over from an earlier one.

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.tools import ComponentTool, flatten_tools_or_toolsets
from multihop_rag import CORPUS_KEY, SPLIT_LENGTH, SPLIT_OVERLAP, build_eval_cases, prepare_corpus
from util import build_bm25_retriever

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent, prompts
from haystack_integrations.agent_pack.advanced_rag.harness_evaluator import (
    AdvancedRAGHarnessEvaluator,
)
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    ExperimentResult,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
)
from haystack_integrations.agent_pack.optimization.prompts import OPTIMIZER_PROMPT_CACHE_KEY
from haystack_integrations.evaluation import RAGEvalCase, ToolNames
from haystack_integrations.evaluation.agent_run_digest import AgentRunDigestPolicy

WORKSPACE = Path(".agent-pack-poc")
REFERENCE_MODEL = "gpt-5.6-luna"
CANDIDATE_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")

# What only this harness knows about its own reference Agent. It describes the Agent and how its retrieval is
# measured, so the optimizer does not have to infer all of that from serialized class names, and deliberately
# prescribes no fix: which limit to change, and to what, is what the experiment is for. General experiment
# discipline is not repeated here — it belongs to every harness and lives in the optimizer instructions.
# A metadata listing is only usable as evidence when the optimizer knows it is complete: these three tools answer
# "what values exist", and a truncated answer invites a candidate that hard-codes an incomplete set.
DIGEST_POLICY = AgentRunDigestPolicy(
    keep_full_results_for=frozenset({"list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"})
)

ADVANCED_RAG_OPTIMIZER_GUIDANCE = """
The reference Agent is an Advanced RAG coordinator. It inspects document-store metadata, builds a Haystack metadata
filter from what it finds, retrieves with that filter, and cites the documents it used.

Retrieval has two independent paths with separate limits. `search_documents` ranks by relevance and returns at most
the retriever's own `top_k`. `fetch_documents_by_filter` returns an exact filtered set and refuses outright when the
filter matches more documents than its own per-fetch limit allows. Evaluation cases state how many matching documents
an answer needs, some require the complete filtered set, and each eval case budgets its metadata and retrieval calls.

An eval case can require evidence that is spread across several documents, and one query phrased for the whole question
will tend to surface the documents that share its wording and miss the rest. A retrieval budget therefore does not
have to be spent on one broad search: it can be spent on one search per piece of evidence the question asks for,
each phrased for that piece. Retrieving too little and retrieving loosely are separate failures, and the eval case
reports which one occurred.

The retrieval tool itself is part of the configuration and can be replaced, not only retuned. It is a single
keyword retriever, which ranks by wording alone, and the way to improve it here is to make one search better
rather than to make the tool issue several. This Agent already searches more than once: decomposing a question
into a search per piece of evidence is something its instructions can ask for, and every call it makes is
separately ranked against its own query. A query expander inside the tool would repeat that at a second level
while the Agent keeps doing it at the first.

What one search returns is therefore the thing worth improving, and a ranker behind the retriever is what improves
it: the retriever can consider a wide candidate set on wording, and the ranker can pick from it on whether a
document adds a facet the question asks for.

Those are two separate numbers and only the second one is what the tool returns. Let the retriever consider plenty
and have the ranker cut it to roughly five to seven documents per search. Three is too few even when the ranking
is good, since an eval case needs at least three matching documents and a single search rarely holds all of them.
Past seven the cost is where it shows: everything returned enters the Agent's context and stays there for every
later step, so width is paid for on every subsequent call as input tokens rather than as a quality failure. Read
the reported cost against the recall to find where widening stopped buying evidence and started buying context.

Retrieval that keeps failing once both the instructions and the retriever's own limits have been tuned is evidence
about that mechanism rather than about the wording of either, and the mechanism is then the variable worth a
measurement.

Eval cases going over their search budget are the clearest sign of that. An Agent searches again because the last
search did not return what it needed, so a run that spends two or three times its allowance is reporting that each
individual search is too weak, not that the Agent is undisciplined. Tightening the instructions to search less
makes the budget failure go away while leaving the evidence unfound; strengthening what one search returns fixes
both. Read the over-budget eval cases together with the recall on them before deciding which of the two you are
looking at. Confirm what such a pipeline serializes to, and which
components this environment can actually import, before spending one on it.
A tool wrapping a single component can become a tool wrapping a pipeline. To put a ranker behind this Agent's
retrieval tool, build a PipelineTool: connect retriever.documents to ranker.documents, map the tool's query to
both query inputs, map filters to the retriever, and expose ranker.documents as the tool's documents output.
Whatever replaces the tool has to keep the outputs_to_state mappings and formatting handlers the harness reads.

Quality is the share of each eval case's needed documents that the answer actually cited, averaged over the
eval cases. Retrieving the evidence is necessary but not sufficient: a run that retrieves everything and cites
none of it scores nothing. The retrieval recall is reported beside it, so the two are distinguishable — retrieval
that is already complete points at the answer step rather than at the retriever.

A tool that comes back an error fails the eval case without voiding what the run retrieved, and the count is
reported per eval case. Errors concentrated in one tool are evidence about the configuration rather than about the
question: a filter built against a field that does not exist, a limit the tool refuses, a tool that can never
return anything. Read which tool errored and why, and change the configuration so it stops rather than budgeting
for it.

The model a generator runs on is part of the configuration, and `gpt-5.6-terra` is available for any of them.
Moving one is a real variable in either direction: what it costs is priced and reported, and whether the
decomposition and citation decisions a model makes are worth what it charges for them is exactly the kind of
question a measurement answers. Read the configuration for what each generator is on now rather than assuming.
Configuration and prompt repairs are usually the cheaper fix and worth trying first, but a quality objective that
has stopped moving on those is a reason to measure the model rather than to keep rewriting instructions.

`gpt-5.6-sol` is the one exception: it is out of scope for this experiment, so do not move any component onto
it and do not propose it as a change worth measuring.
""".strip()

# The reference Agent starts badly configured on both axes the experiment measures, so there is real ground for the
# optimizer to gain. Quality: retrieval is starved from both sides, because `search_documents` returns a single
# document and a filter fetch shows two, while every eval case demands at least three matching documents; the
# loop is then cut off after a few steps, so a run that does retrieve is liable to be summarized by the
# backup-answer hook without citations. Cost: it carries a leftover retrieval tool whose schema is sent to the
# model on every step and which can never return anything (see `build_leftover_tool`).
#
# The reference starts on the cheapest model, so the optimizer cannot buy its improvement by downgrading. A broken
# configuration wastes money flailing — measured: a starved reference spent 19 retrieval calls and 37,011 input
# tokens across its eval cases, and repairing it on the same model needed 3 calls and 24,955 tokens, 42% cheaper — so a
# quality repair still clears the cost objective here, and it has to come from the configuration rather than the
# price list.
POOR_RETRIEVER_TOP_K = 1
POOR_LEFTOVER_TOOL_NAME = "search_product_manuals"
POOR_MAX_FETCHED_DOCS = 2
POOR_MAX_AGENT_STEPS = 6
REFERENCE_REASONING_EFFORT = "low"

# What an eval case will pay for before it is voided. An answer here needs up to four documents spread across
# separate articles, and one search per piece of evidence is what finds them, so the retrieval allowance has to
# leave room for that plus a retry rather than capping the decomposition the instructions ask for. Metadata is
# inspected once per field and not per piece of evidence, so it needs less.
TOOL_BUDGETS: dict[ToolNames, int] = {
    ("search_documents", "fetch_documents_by_filter"): 10,
    ("list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"): 6,
}

# USD prices per million tokens, used only to rank candidates against each other.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}


def build_optimizer_generator(model: str | None) -> OpenAIResponsesChatGenerator | None:
    """
    Build the optimizer's generator on a named model, leaving everything else as the default.

    Only the model differs from what the library would build, so a run that changes it measures the model rather
    than the settings around it.

    :param model: Model to reason with, or None to accept the library default.
    :returns: The generator, or None to let `create_harness_optimizer_agent` choose.
    """
    if model is None:
        return None
    return OpenAIResponsesChatGenerator(
        model=model,
        timeout=180.0,
        max_retries=5,
        generation_kwargs={"prompt_cache_key": OPTIMIZER_PROMPT_CACHE_KEY, "reasoning": {"effort": "low"}},
    )


def build_leftover_tool() -> ComponentTool:
    """
    Build a retrieval tool left over from another corpus, pointing at a store that holds nothing.

    Nothing any evaluation case asks for is in there, so the tool can only ever return nothing — but its name,
    description and full argument schema are sent to the model on every step regardless. It is the kind of tool
    that accumulates in a configuration nobody prunes, and removing it costs no quality.

    Its schema is spelled out rather than derived, and carries the same filter grammar the real retrieval tool
    does, because that grammar is what makes such a tool expensive: the cost of keeping it is paid per model call,
    and a tool whose schema is a couple of hundred characters is not worth an experiment to remove.

    :returns: The useless tool.
    """
    return ComponentTool(
        component=InMemoryBM25Retriever(document_store=InMemoryDocumentStore()),
        name=POOR_LEFTOVER_TOOL_NAME,
        description=(
            "Search the product manual corpus for troubleshooting steps, specifications, warranty terms and "
            "service intervals. Use it when a question concerns how a product is meant to be used, installed or "
            "serviced rather than what reviewers said about it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The manual search query: a short phrase or a few keywords describing the procedure, "
                        "specification or warranty clause to retrieve."
                    ),
                },
                "filters": {
                    "type": "object",
                    "description": prompts.FILTER_GRAMMAR,
                    "additionalProperties": True,
                },
            },
            "required": ["query"],
        },
    )


def build_reference_agent(store: DocumentStore, model: str) -> Agent:
    """Build the badly configured Advanced RAG Agent whose complete configuration will be optimized."""
    generation_kwargs = {"reasoning": {"effort": REFERENCE_REASONING_EFFORT}}
    agent = create_advanced_rag_agent(
        document_store=store,
        retriever=build_bm25_retriever(store=store, top_k=POOR_RETRIEVER_TOP_K),
        llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs=generation_kwargs),
        # Keep backup-answer usage attributable to the selected reference model. Changing only the coordinator's
        # model path leaves this fallback unchanged unless the optimizer explicitly edits it too.
        backup_answer_llm=OpenAIResponsesChatGenerator(model=model, generation_kwargs=generation_kwargs),
        max_agent_steps=POOR_MAX_AGENT_STEPS,
        max_fetched_docs=POOR_MAX_FETCHED_DOCS,
    )
    return agent.clone(tools=[*agent.tools, build_leftover_tool()])


def build_pricing(models: tuple[str, ...]) -> ModelPriceCatalog:
    """Build informational prices for the models currently known to this example."""
    return ModelPriceCatalog(
        prices=[
            ModelPrice(
                model_id=model,
                input_cost_per_million=MODEL_PRICES[model][0],
                output_cost_per_million=MODEL_PRICES[model][1],
            )
            # Deduplicated because the reference model is usually also one of the candidates, and a catalog rejects
            # a repeated identifier.
            for model in dict.fromkeys(models)
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
    for eval_case_metrics in baseline.details.get("eval_cases", []):
        verdict = "pass" if eval_case_metrics["passed"] else "FAIL " + ",".join(eval_case_metrics["failures"])
        print(f"    [{verdict}] {eval_case_metrics['question']}")

    print("\n--- candidates ---")
    for candidate in result.candidates:
        gates = result.gate_failures.get(candidate.candidate_id, ())
        if candidate.metrics is None:
            print(f"  {candidate.candidate_id} -> failed: {candidate.failure}")
            continue
        print(
            f"  {candidate.candidate_id} -> quality={candidate.metrics.quality:.2f} "
            f"cost={format_cost(cost=candidate.metrics.cost)} latency={candidate.metrics.latency_ms:.0f}ms"
        )
        print(f"    gates: {'passed' if not gates else ', '.join(gates)}")
        for eval_case_metrics in candidate.metrics.details.get("eval_cases", []):
            if not eval_case_metrics["passed"]:
                print(
                    f"      regression on {eval_case_metrics['question']!r}: {','.join(eval_case_metrics['failures'])}"
                )

    print("\n--- what the search itself cost ---")
    usage = ", ".join(
        f"{model}: {tokens.input_tokens} in / {tokens.output_tokens} out"
        for model, tokens in result.optimizer_usage.items()
    )
    print(f"  optimizer usage: {usage or 'none recorded'}")
    print(f"  optimizer cost:  {format_cost(cost=result.optimizer_cost)} (input tokens charged at full price)")

    print("\n--- recommendation ---")
    if result.recommendation is None:
        print("  none: no candidate cleared every gate and improved on the reference")
        return
    recommendation = result.recommendation
    print(f"  configuration: {result.artifact_directory / 'recommended.yaml'}")
    print(f"  rationale: {recommendation.configuration.rationale}")
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
    parser.add_argument("--workspace", type=Path, default=WORKSPACE, help="Directory for runs and YAML artifacts.")
    parser.add_argument("--candidate-model", action="append", dest="candidate_models")
    parser.add_argument(
        "--config", type=Path, help="Optional editable YAML file; created from the reference if absent."
    )
    parser.add_argument(
        "--max-eval-cases",
        type=int,
        default=20,
        help="Eval cases to evaluate. Quality is a fraction of these, so fewer of them make it a coarser "
        "measurement, while each one costs an Agent run for every candidate measured.",
    )
    parser.add_argument(
        "--eval-case-seed",
        type=int,
        default=0,
        help="Selects which eval cases are drawn from the dataset. The same seed rebuilds the same evaluation set.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=1.0,
        help="Minimum candidate eval case pass rate in the inclusive range 0.0 to 1.0.",
    )
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.0,
        help="Maximum absolute pass-rate loss from the reference, between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--primary",
        choices=("cost", "latency", "quality"),
        default="cost",
        help="What candidates are ranked by. `cost` and `latency` are minimized among candidates that clear the "
        "quality gates; `quality` is maximized directly with cost breaking ties, which needs no quality floor to "
        "be chosen in advance.",
    )
    parser.add_argument(
        "--max-concurrent-eval-cases",
        type=int,
        default=4,
        help="How many eval cases to measure at once. They are independent and spend their time waiting on a "
        "model, so this is what decides how long an experiment takes. It does not change token usage, but "
        "concurrent runs contend for rate limits, so it must be 1 when ranking by latency.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of candidate configurations included in the experiment.",
    )
    parser.add_argument(
        "--optimizer-model",
        help="Model the optimizer itself reasons with. Defaults to whatever `create_harness_optimizer_agent` chooses.",
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


def enable_progress_reporting() -> None:
    """
    Report progress while the experiment runs rather than when it finishes.

    A run spends most of its time inside Agent calls, and its own output is a few lines per phase, so a redirected
    stdout would otherwise stay empty for the length of an experiment: line buffering makes each line appear as it
    is written. The library reports each eval case and each candidate through its logger, which is routed here so that
    progress and phases arrive on the same stream in order.
    """
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("  %(message)s"))
    progress = logging.getLogger("haystack_integrations.agent_pack")
    progress.handlers.clear()
    progress.addHandler(handler)
    progress.setLevel(logging.INFO)
    progress.propagate = False


def main() -> None:
    """Build the experiment and run it end to end."""
    arguments = parse_args()
    enable_progress_reporting()
    if not os.environ.get("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY must be set to run this walkthrough."
        raise SystemExit(message)
    if arguments.max_eval_cases < 1:
        message = "--max-eval-cases must be at least 1."
        raise SystemExit(message)
    if arguments.max_concurrent_eval_cases < 1:
        message = "--max-concurrent-eval-cases must be at least 1."
        raise SystemExit(message)
    if arguments.primary == "latency" and arguments.max_concurrent_eval_cases > 1:
        message = (
            "Ranking by latency requires --max-concurrent-eval-cases 1: concurrent runs contend for the same rate "
            "limits, so the measurement would describe that contention rather than the configuration."
        )
        raise SystemExit(message)
    if arguments.fresh and arguments.workspace.exists():
        shutil.rmtree(path=arguments.workspace)

    print("=== 1. set up corpus and evaluation set ===")
    store, articles = prepare_corpus(backend=arguments.store)
    document_count = store.count_documents()
    print(f"  {CORPUS_KEY} on {arguments.store}: {document_count} chunks from {len(articles)} articles")

    # The dataset reports what it labels; turning that into what this harness scores is the harness's own call.
    # The ground-truth answer is left out: many are short words like "Yes", where a substring check on a reply
    # can match for reasons unrelated to the answer being right.
    labelled = build_eval_cases(articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed)
    eval_cases = [
        RAGEvalCase(question=question.question, evidence=question.evidence, tool_budgets=TOOL_BUDGETS)
        for question in labelled
    ]
    print(f"  eval cases: {len(eval_cases)} labelled from evidence")
    candidate_models = tuple(arguments.candidate_models or CANDIDATE_MODELS)
    reference_agent = build_reference_agent(store=store, model=arguments.reference_model)
    tool_names = sorted(configured.name for configured in flatten_tools_or_toolsets(tools=reference_agent.tools))
    print(f"  model={arguments.reference_model} tools={tool_names}")

    pricing = build_pricing(models=(arguments.reference_model, *candidate_models))

    print("\n=== 2. optimization experiment ===")
    experiment = HarnessOptimizationExperiment(
        reference=reference_agent,
        eval_cases=eval_cases,
        evaluator=AdvancedRAGHarnessEvaluator(
            digest_policy=DIGEST_POLICY,
            max_concurrent_eval_cases=arguments.max_concurrent_eval_cases,
        ),
        pricing=pricing,
        objectives=OptimizationObjectives(
            min_quality=arguments.min_quality,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=ExperimentJournal(directory=arguments.workspace / "journals"),
        optimizer_agent=create_harness_optimizer_agent(
            chat_generator=build_optimizer_generator(model=arguments.optimizer_model),
            documentation_tools=arguments.docs_mcp,
            additional_instructions=ADVANCED_RAG_OPTIMIZER_GUIDANCE,
        ),
        max_iterations=arguments.max_iterations,
        config_path=arguments.config,
        configuration_key=f"{CORPUS_KEY}:{SPLIT_LENGTH}:{SPLIT_OVERLAP}:{document_count}",
    )
    result = experiment.run()
    print(f"  measurement context: {result.measurement_context}; run: {result.run_id}")

    print("\n=== 3. outcome ===")
    report(result=result)
    print(f"\nJournal: {experiment.journal.path_for(result.run_id)}")


if __name__ == "__main__":
    main()
