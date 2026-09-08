# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def load_structured_data_prompts() -> dict[str, str]:
    """Load the exact structured-data prompt presets bundled with the package."""
    resource = files("haystack_integrations.utils.mrscraper").joinpath("structured_data_prompts.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in data.items()
    ):
        msg = "The bundled MrScraper structured-data prompts are invalid."
        raise RuntimeError(msg)
    return data
