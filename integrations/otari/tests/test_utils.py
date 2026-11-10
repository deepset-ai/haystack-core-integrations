# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.common.otari.utils import _warn_if_eu_key_on_default_url


def test_warns_about_eu_key_on_default_url(caplog):
    _warn_if_eu_key_on_default_url(Secret.from_token("otk_v1_eu_test"), "https://api.otari.ai/api/v1")
    assert "belongs to otari.ai's EU region" in caplog.text
    assert "https://eu.api.otari.ai/api/v1" in caplog.text


@pytest.mark.parametrize(
    ("api_key", "api_base_url"),
    [
        ("otk_v1_eu_test", "https://eu.api.otari.ai/api/v1"),
        ("otk_v1_eu_test", "http://localhost:8000/api/v1"),
        ("otk_v1_us_test", "https://api.otari.ai/api/v1"),
        ("tk-local-gateway-key", "https://api.otari.ai/api/v1"),
    ],
)
def test_does_not_warn_otherwise(caplog, api_key, api_base_url):
    _warn_if_eu_key_on_default_url(Secret.from_token(api_key), api_base_url)
    assert "EU region" not in caplog.text
