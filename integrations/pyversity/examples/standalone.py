# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from pyversity import Strategy

from haystack_integrations.components.rankers.pyversity import PyversityRanker

documents = [
    Document(content="Paris is the capital of France.", score=0.95, embedding=[0.9, 0.1, 0.0, 0.0]),
    Document(content="The Eiffel Tower is located in Paris.", score=0.90, embedding=[0.8, 0.2, 0.0, 0.0]),
    Document(content="Berlin is the capital of Germany.", score=0.85, embedding=[0.0, 0.0, 0.9, 0.1]),
    Document(content="The Brandenburg Gate is in Berlin.", score=0.80, embedding=[0.0, 0.0, 0.8, 0.2]),
    Document(content="France borders Spain to the south.", score=0.75, embedding=[0.5, 0.5, 0.0, 0.0]),
]

reranker = PyversityRanker(top_k=3, strategy=Strategy.MMR, diversity=0.7)
result = reranker.run(documents=documents)

for doc in result["documents"]:
    print(f"{doc.score:.2f}  {doc.content}")

# 0.28  Paris is the capital of France.
# 0.26  Berlin is the capital of Germany.
# -0.32  France borders Spain to the south.
