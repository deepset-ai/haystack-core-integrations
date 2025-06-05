# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .document_embedder import GoogleAIDocumentEmbedder
from .text_embedder import GoogleAITextEmbedder

__all__ = ["GoogleAIDocumentEmbedder", "GoogleAITextEmbedder"]
