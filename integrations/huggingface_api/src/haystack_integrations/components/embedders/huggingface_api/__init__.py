# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .document_embedder import HuggingFaceAPIDocumentEmbedder
from .text_embedder import HuggingFaceAPITextEmbedder

__all__ = ["HuggingFaceAPIDocumentEmbedder", "HuggingFaceAPITextEmbedder"]
