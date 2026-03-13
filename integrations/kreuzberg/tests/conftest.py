# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from kreuzberg import ExtractionConfig, PageConfig

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter


@pytest.fixture
def sequential_converter():
    return KreuzbergConverter(batch=False)


@pytest.fixture
def per_page_converter():
    return KreuzbergConverter(
        config=ExtractionConfig(pages=PageConfig(extract_pages=True)),
        batch=False,
    )
