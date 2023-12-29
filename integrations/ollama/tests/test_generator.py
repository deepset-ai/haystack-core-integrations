# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from requests import HTTPError

from ollama_haystack import OllamaGenerator


class TestOllamaGenerator:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "prompt, answer",
        [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ],
    )
    def test_run_capital_cities(self, prompt, answer):
        component = OllamaGenerator()
        results = component.run(prompt=prompt)
        response = results["replies"][0]
        assert answer in response

    @pytest.mark.integration
    def test_run_model_unavailable(self):
        component = OllamaGenerator(model_name="Alistair_is_great")

        with pytest.raises(HTTPError):
            component.run(prompt="Why is Alistair so great?")

    def test__get_telemetry_data(self):
        component = OllamaGenerator(model_name="some_model")
        observed = component._get_telemetry_data()
        assert observed == {"model": "some_model"}

    @pytest.mark.parametrize(
        "configuration, prompt",
        [
            (
                {
                    "model_name": "some_model",
                    "url": "https://localhost:11434/api/generate",
                    "raw": True,
                    "system_prompt": "You are mario from Super Mario Bros.",
                    "template": None,
                },
                "hello",
            ),
            (
                {
                    "model_name": "some_model2",
                    "url": "https://localhost:11434/api/generate",
                    "raw": False,
                    "system_prompt": None,
                    "template": "some template",
                },
                "hello",
            ),
        ],
    )
    def test__post_args(self, configuration, prompt):
        component = OllamaGenerator(**configuration)

        observed = component._post_args(prompt=prompt)

        expected = {
            "url": "https://localhost:11434/api/generate",
            "timeout": 30,
            "json": {
                "prompt": prompt,
                "model": configuration["model_name"],
                "stream": False,
                "system": configuration['system_prompt'],
                "raw": configuration["raw"],
                "template": configuration['template'],
                "options": {},
            },
        }

        assert observed == expected
