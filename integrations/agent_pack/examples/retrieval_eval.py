# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Score a retrieval pipeline against the MultiHopRAG evaluation set.
#
# The corpus and its labelled questions come from `multihop_rag`, which chunks the news articles and works out
# which chunks hold each question's quoted evidence. That gives exact ground truth: an eval case contains the
# required documents to answer a question, so retrieval is scored as recall@k over those IDs.
#
# `RetrievalHarnessEvaluator` drives any pipeline that takes a `query` and returns `documents`. It auto-detects those
# sockets on the provided pipeline.
#
# The pipeline here expands the question into several queries with an LLM, retrieves for each and pools the
# results.
#
# Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:
#
#     hatch run test:python examples/retrieval_eval.py
#     hatch run test:python examples/retrieval_eval.py --max-eval-cases 20 --k 5

import argparse

from haystack import Pipeline
from haystack.components.generators.chat.openai_responses import OpenAIResponsesChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.retrievers import MultiQueryTextRetriever
from haystack.document_stores.types import DocumentStore
from multihop_rag import CORPUS_KEY, build_eval_cases, prepare_corpus
from util import build_bm25_retriever, preview

from haystack_integrations.evaluation import RetrievalEvalCase, RetrievalHarnessEvaluator

# Expansion schema for the QueryExpander component
EXPANSION_SCHEMA = {
    "type": "json_schema",
    "name": "expanded_queries",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
        "required": ["queries"],
        "additionalProperties": False,
    },
}


def parse_args() -> argparse.Namespace:
    """Read the evaluation set size, the rank cutoff and how the question is expanded."""
    parser = argparse.ArgumentParser(description="Score a retrieval pipeline on the MultiHopRAG evaluation set.")
    parser.add_argument("--max-eval-cases", type=int, default=10, help="How many labelled questions to score.")
    parser.add_argument("--eval-case-seed", type=int, default=0, help="Selects which eval cases are drawn.")
    parser.add_argument("--k", type=int, default=10, help="Rank cutoff, giving recall@k and precision@k.")
    parser.add_argument("--top-k", type=int, default=10, help="How many documents one retrieval returns.")
    parser.add_argument("--expansions", type=int, default=3, help="How many extra queries to expand into.")
    parser.add_argument("--model", default="gpt-5.4", help="The model that expands the question.")
    parser.add_argument("--concurrency", type=int, default=4, help="How many eval cases to measure at once.")
    return parser.parse_args()


def build_pipeline(store: DocumentStore, arguments: argparse.Namespace) -> Pipeline:
    """
    Build the pipeline to score: an LLM query expansion pooled over one BM25 retrieval per query.

    :param store: The corpus to retrieve from.
    :param arguments: The parsed command line, giving the expansion count, model and retrieval depth.
    :returns: A pipeline exposing a `query` input and a `documents` output, which is all the harness needs.
    """
    retriever = build_bm25_retriever(store=store, top_k=arguments.top_k)
    pipeline = Pipeline()
    expander = QueryExpander(
        chat_generator=OpenAIResponsesChatGenerator(
            model=arguments.model,
            generation_kwargs={"reasoning": {"effort": "low"}, "text": {"format": EXPANSION_SCHEMA}},
        ),
        n_expansions=arguments.expansions,
    )
    pipeline.add_component("expander", expander)
    # One retrieval per expanded query, pooled and ranked by score.
    pipeline.add_component("retriever", MultiQueryTextRetriever(retriever=retriever))
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def main() -> None:
    """Build the corpus, score a retrieval pipeline against it, and print what each eval case cost and did."""
    arguments = parse_args()

    print("=== 1. set up corpus and evaluation set ===")
    store, articles = prepare_corpus(backend="in_memory")
    chunks = sum(len(article.chunks) for article in articles.values())
    print(f"  {CORPUS_KEY} in memory: {chunks} chunks from {len(articles)} articles")

    labelled = build_eval_cases(articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed)
    eval_cases = [RetrievalEvalCase(question=one.question, evidence=one.evidence) for one in labelled]
    print(f"  eval cases: {len(eval_cases)} labelled from evidence, scored at recall@{arguments.k}")

    # Any pipeline exposing a `query` input and a `documents` output can be scored, whatever runs in between.
    pipeline = build_pipeline(store=store, arguments=arguments)

    print("\n=== 2. evaluate ===")
    print(f"  pipeline: {' -> '.join(pipeline.graph.nodes)}")
    print(
        f"  {arguments.model} expanding into {arguments.expansions} extra queries, "
        f"{arguments.top_k} documents retrieved per query"
    )
    evaluator = RetrievalHarnessEvaluator(k=arguments.k, max_concurrent_eval_cases=arguments.concurrency)
    evaluator.validate(target=pipeline)
    metrics = evaluator.evaluate(target=pipeline, eval_cases=eval_cases)

    print("\n=== 3. outcome ===")
    print(
        f"  quality={metrics.quality:.2f} (mean recall@{arguments.k})  "
        f"recall@k={metrics.details['mean_recall_at_k']:.2f}  "
        f"precision@k={metrics.details['mean_precision_at_k']:.2f}  "
        f"latency={metrics.latency_ms:.0f}ms"
    )
    # How much each stage emitted, which no configuration value states once a pipeline pools or deduplicates.
    for component, sockets in metrics.details["stage_output_sizes"].items():
        for socket, sizes in sockets.items():
            print(f"    {component}.{socket}: {sizes['median']} typical ({sizes['min']}-{sizes['max']})")

    # What the retrieval cost, gathered from the spans the pipeline's model calls emitted.
    for model, usage in metrics.model_usage.items():
        per_eval_case = (usage.input_tokens + usage.output_tokens) / len(eval_cases)
        print(f"    {model}: {usage.input_tokens} in + {usage.output_tokens} out ({per_eval_case:.0f} per eval case)")
    if metrics.model_usage and not metrics.details["all_tokens_reported"]:
        print("    (a model call reported no token counts, so the totals above are an undercount)")

    quotes = {document: quote for eval_case in eval_cases for document, quote in eval_case.evidence.items()}
    for position, reported in enumerate(metrics.details["eval_cases"], start=1):
        needed = len(eval_cases[position - 1].evidence)
        missed = reported["missed_document_ids"]
        print(f"\n=== eval case {position}/{len(eval_cases)}: {'PASS' if reported['passed'] else 'FAIL'} ===")
        print(f"  question: {reported['question']}")
        print(
            f"  retrieval: found {needed - len(missed)}/{needed} of the documents the answer needs, "
            f"{reported['retrieved']} returned"
        )
        print(
            f"  recall@{arguments.k}: {reported['recall_at_k']:.2f}   precision@{arguments.k}: "
            f"{reported['precision_at_k']:.2f}   time: {reported['latency_ms']:.0f}ms"
        )
        # What the pipeline actually put to the store, under the same names section 3 counted. A recall
        # failure is usually explained by the queries rather than by how many of them there were.
        for component, sockets in reported["stage_texts"].items():
            for socket, texts in sockets.items():
                print(f"  {component}.{socket}:")
                for text in texts:
                    print(f"    {preview(text=text, limit=108)}")
        for document_id in missed:
            print(
                f"  needed but never retrieved: [doc {document_id[:8]}] {preview(text=quotes[document_id], limit=96)}"
            )

    for warning in metrics.details["warnings"]:
        print(f"  warning x{warning['count']}: {warning['message'][:100]}")


if __name__ == "__main__":
    main()
