# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods
from unittest.mock import Mock

from haystack_integrations.components.generators.openvino import OpenVINOGenerator


class TestOpenVINOGenerator:
    def test_init_default(self):
        generator = OpenVINOGenerator()

        assert generator.huggingface_pipeline_kwargs == {
            "model": "microsoft/Phi-3-mini-4k-instruct",
            "task": "text-generation",
            "device": "cpu",
        }
        assert generator.generation_kwargs == {"max_new_tokens": 512}
        assert generator.pipeline is None

    def test_warm_up(self, pipeline_mock):
        generator = OpenVINOGenerator(model="microsoft/Phi-3-mini-4k-instruct", task="text-generation")
        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            device="cpu",
        )

    def test_run(self):
        generator = OpenVINOGenerator(
            model="microsoft/Phi-3-mini-4k-instruct", task="text-generation", generation_kwargs={"max_new_tokens": 100}
        )

        # create the pipeline object (simulating the warm_up)
        generator.pipeline = Mock(return_value=[{"generated_text": "Rome"}])

        results = generator.run(prompt="What's the capital of Italy?")

        generator.pipeline.assert_called_once_with(
            "What's the capital of Italy?", max_new_tokens=100, stopping_criteria=None
        )
        assert results == {"replies": ["Rome"]}
