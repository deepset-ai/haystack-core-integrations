# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.components.embedders.perplexity.document_embedder import (
    PerplexityDocumentEmbedder,
)
from haystack_integrations.components.embedders.perplexity.text_embedder import (
    PerplexityTextEmbedder,
)

__all__ = ["PerplexityDocumentEmbedder", "PerplexityTextEmbedder"]
