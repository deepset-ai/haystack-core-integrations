# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from ollama_haystack.generator import OllamaGenerator

class TestOllamaGenerator:

    @pytest.mark.integration
    def test_ollama_generator_run(self):
        component = OllamaGenerator()
        results = component.run(prompt="What's the capital of France?")
        response = results["replies"]
        assert "Paris" in response