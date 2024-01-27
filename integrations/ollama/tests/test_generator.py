# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import StreamingChunk
from haystack_integrations.components.generators.ollama import OllamaGenerator
from requests import HTTPError


class TestOllamaGenerator:
    @pytest.mark.integration
    def test_run_capital_cities(self):
        prompts_and_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        component = OllamaGenerator()

        for prompt, answer in prompts_and_answers:
            results = component.run(prompt=prompt)
            response = results["replies"][0]

            assert "replies" in results
            assert "meta" in results
            assert answer in response

    @pytest.mark.integration
    def test_run_model_unavailable(self):
        component = OllamaGenerator(model="Alistair_is_great")

        with pytest.raises(HTTPError):
            component.run(prompt="Why is Alistair so great?")

    def test_init_default(self):
        component = OllamaGenerator()
        assert component.model == "orca-mini"
        assert component.url == "http://localhost:11434/api/generate"
        assert component.generation_kwargs == {}
        assert component.system_prompt is None
        assert component.template is None
        assert component.raw is False
        assert component.timeout == 120
        assert component.streaming_callback is None

    def test_init(self):
        def callback(x: StreamingChunk):
            x.content = ""

        component = OllamaGenerator(
            model="llama2",
            url="http://my-custom-endpoint:11434/api/generate",
            generation_kwargs={"temperature": 0.5},
            system_prompt="You are Luigi from Super Mario Bros.",
            timeout=5,
            streaming_callback=callback,
        )
        assert component.model == "llama2"
        assert component.url == "http://my-custom-endpoint:11434/api/generate"
        assert component.generation_kwargs == {"temperature": 0.5}
        assert component.system_prompt == "You are Luigi from Super Mario Bros."
        assert component.template is None
        assert component.raw is False
        assert component.timeout == 5
        assert component.streaming_callback == callback

    @pytest.mark.parametrize(
        "configuration",
        [
            {
                "model": "some_model",
                "url": "https://localhost:11434/api/generate",
                "raw": True,
                "system_prompt": "You are mario from Super Mario Bros.",
                "template": None,
            },
            {
                "model": "some_model2",
                "url": "https://localhost:11434/api/generate",
                "raw": False,
                "system_prompt": None,
                "template": "some template",
            },
        ],
    )
    @pytest.mark.parametrize("stream", [True, False])
    def test_create_json_payload(self, configuration, stream):
        prompt = "hello"
        component = OllamaGenerator(**configuration)

        observed = component._create_json_payload(prompt=prompt, stream=stream)

        expected = {
            "prompt": prompt,
            "model": configuration["model"],
            "stream": stream,
            "system": configuration["system_prompt"],
            "raw": configuration["raw"],
            "template": configuration["template"],
            "options": {},
        }

        assert observed == expected


class TestOllamaStreamingGenerator:
    @pytest.mark.integration
    def test_ollama_generator_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.count_calls = 0

            def __call__(self, chunk):
                self.responses += chunk.content
                self.count_calls += 1
                return chunk

        callback = Callback()
        component = OllamaGenerator(streaming_callback=callback)
        results = component.run(prompt="What's the capital of Netherlands?")

        assert len(results["replies"]) == 1
        assert "Amsterdam" in results["replies"][0]
        assert len(results["meta"]) == 1
        assert callback.responses == results["replies"][0]
        assert callback.count_calls > 1
