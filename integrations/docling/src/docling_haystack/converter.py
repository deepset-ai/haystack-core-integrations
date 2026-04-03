"""Backward-compatibility shim for the old docling-haystack import path."""

import warnings

warnings.warn(
    "Importing from 'docling_haystack.converter' is deprecated and will be removed in a future release. "
    "Use 'from haystack_integrations.components.converters.docling import DoclingConverter' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from haystack_integrations.components.converters.docling.converter import (  # noqa: E402, F401
    BaseMetaExtractor,
    DoclingConverter,
    ExportType,
    MetaExtractor,
)
