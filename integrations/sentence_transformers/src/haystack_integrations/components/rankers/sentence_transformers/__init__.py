# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .sentence_transformers_diversity import SentenceTransformersDiversityRanker
from .sentence_transformers_similarity import SentenceTransformersSimilarityRanker

__all__ = [
    "SentenceTransformersDiversityRanker",
    "SentenceTransformersSimilarityRanker",
]
