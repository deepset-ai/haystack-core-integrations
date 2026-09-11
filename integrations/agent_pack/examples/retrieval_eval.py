# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Score a retrieval pipeline against the MultiHopRAG evaluation set.
#
# The corpus and its labelled questions come from `multihop_rag`, which chunks the news articles and works out
# which chunks hold each question's quoted evidence. That gives exact ground truth: an eval case names the
# documents an answer needs, so retrieval is scored as recall@k over those IDs.
#
# `RetrievalHarnessEvaluator` drives any pipeline that takes a `query` and returns `documents`. It finds those
# sockets by name rather than by component name, so the same harness scores a plain BM25 retriever and a
# multi-stage pipeline without being told how either is wired.
#
# Run from `integrations/agent_pack`. The corpus requires `datasets`:
#
#     hatch run test:python examples/retrieval_eval.py
#     hatch run test:python examples/retrieval_eval.py --max-eval-cases 20 --k 5
#     hatch run test:python examples/retrieval_eval.py --store opensearch
#
# `--store opensearch` reuses a populated index across runs, so repeat runs skip re-indexing. Set `OPENSEARCH_URL`
# if it is not http://localhost:9200, and `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD` for a secured instance.

import argparse

from haystack import Pipeline
from multihop_rag import CORPUS_KEY, build_eval_cases, prepare_corpus
from util import build_bm25_retriever

from haystack_integrations.evaluation import RetrievalEvalCase, RetrievalHarnessEvaluator


def parse_args() -> argparse.Namespace:
    """Read the evaluation set size, the rank cutoff and which store to build the corpus in."""
    parser = argparse.ArgumentParser(description="Score a retrieval pipeline on the MultiHopRAG evaluation set.")
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--max-eval-cases", type=int, default=10, help="How many labelled questions to score.")
    parser.add_argument("--eval-case-seed", type=int, default=0, help="Selects which eval cases are drawn.")
    parser.add_argument("--k", type=int, default=10, help="Rank cutoff, giving recall@k and precision@k.")
    parser.add_argument("--top-k", type=int, default=10, help="How many documents one retrieval returns.")
    return parser.parse_args()


def main() -> None:
    """Build the corpus, score a BM25 pipeline against it, and print what each eval case did."""
    arguments = parse_args()

    print("=== 1. set up corpus and evaluation set ===")
    store, articles = prepare_corpus(backend=arguments.store)
    chunks = sum(len(article.chunks) for article in articles.values())
    print(f"  {CORPUS_KEY} on {arguments.store}: {chunks} chunks from {len(articles)} articles")

    labelled = build_eval_cases(articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed)
    eval_cases = [RetrievalEvalCase(question=one.question, evidence=one.evidence) for one in labelled]
    print(f"  eval cases: {len(eval_cases)} labelled from evidence, scored at recall@{arguments.k}")

    # Any pipeline exposing a `query` input and a `documents` output can be scored; this one is the simplest.
    pipeline = Pipeline()
    pipeline.add_component("retriever", build_bm25_retriever(store=store, top_k=arguments.top_k))

    print("\n=== 2. evaluate ===")
    evaluator = RetrievalHarnessEvaluator(k=arguments.k)
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

    print()
    for eval_case in metrics.details["eval_cases"]:
        verdict = "PASS" if eval_case["passed"] else "FAIL"
        print(f"  {verdict}  recall@k={eval_case['recall_at_k']:.2f}  {eval_case['question'][:70]}")
        for missed in eval_case["missed_document_ids"]:
            print(f"          never retrieved: [doc {missed[:8]}]")

    for warning in metrics.details["warnings"]:
        print(f"  warning x{warning['count']}: {warning['message'][:100]}")


if __name__ == "__main__":
    main()
