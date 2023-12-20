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

    def test__get_telemetry_data(self):
        component = OllamaGenerator(model_name='some_model')
        observed = component._get_telemetry_data()
        assert observed == 'some_model'

    def test__post_args(self):
        url = 'https://localhost:9999'

        component = OllamaGenerator(model_name='some_model', url=url)

        observed = component._post_args(prompt='hello')

        expected = {
            'url': f'{url}/api/generate',
            'json': {
                "prompt": 'hello',
                "model": 'some_model',
                "stream": False,
                "raw": True,
                "options": {},
            },
        }

        assert observed == expected

    def test_run(self):
        assert False
