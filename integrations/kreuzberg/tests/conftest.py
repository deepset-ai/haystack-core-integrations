# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from kreuzberg import ExtractionConfig, ExtractionResult, PageConfig

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "test_files"


@pytest.fixture
def converter():
    return KreuzbergConverter()


@pytest.fixture
def sequential_converter():
    return KreuzbergConverter(batch=False)


@pytest.fixture
def per_page_converter():
    return KreuzbergConverter(
        config=ExtractionConfig(pages=PageConfig(extract_pages=True)),
        batch=False,
    )


@pytest.fixture
def make_mock_result():
    """Factory fixture to create mock ExtractionResult instances with realistic defaults.

    Fields that are never ``None`` at runtime (``metadata``, ``tables``,
    ``processing_warnings``, ``output_format``, ``result_format``,
    ``mime_type``) use their actual default values.  Nullable fields
    default to ``None``.

    The metadata dict mirrors real kreuzberg behaviour: it always
    includes output_format and quality_score (when not None).
    """

    def _factory(**overrides: Any) -> MagicMock:
        result = MagicMock(spec=ExtractionResult)
        defaults: dict[str, Any] = {
            "content": "",
            "metadata": {},
            "quality_score": None,
            "processing_warnings": [],
            "detected_languages": None,
            "extracted_keywords": None,
            "output_format": "plain",
            "result_format": "unified",
            "mime_type": "text/plain",
            "tables": [],
            "images": None,
            "annotations": None,
            "pages": None,
            "chunks": None,
        }
        for key, default in defaults.items():
            setattr(result, key, overrides.get(key, default))

        # Mirror real kreuzberg: metadata always includes output_format and
        # quality_score (when not None), matching the PyO3 struct behaviour.
        effective_meta = dict(result.metadata)
        effective_meta.setdefault("output_format", result.output_format)
        if result.quality_score is not None:
            effective_meta.setdefault("quality_score", result.quality_score)
        result.metadata = effective_meta

        return result

    return _factory
