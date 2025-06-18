# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .document_embedder import GoogleGenAIDocumentEmbedder
from .text_embedder import GoogleGenAITextEmbedder

__all__ = ["GoogleGenAIDocumentEmbedder", "GoogleGenAITextEmbedder"]
