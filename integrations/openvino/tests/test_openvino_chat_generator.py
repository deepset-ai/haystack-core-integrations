# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from haystack.dataclasses import ChatMessage, ChatRole
from haystack_integrations.components.generators.openvino import OpenVINOChatGenerator


class TestOpenVINOChatGenerator:
    def test_init_task_parameter(self):
        generator = OpenVINOChatGenerator(device="cpu")

        assert generator.huggingface_pipeline_kwargs == {
            "model": "microsoft/Phi-3-mini-4k-instruct",
            "device": "cpu",
        }

    @patch("haystack.components.generators.chat.hugging_face_local.pipeline")
    def test_warm_up(self, pipeline_mock):
        generator = OpenVINOChatGenerator(
            model="microsoft/Phi-3-mini-4k-instruct",
            device="cpu",
        )

        pipeline_mock.assert_not_called()

        generator.warm_up()

        pipeline_mock.assert_called_once_with(
            model="microsoft/Phi-3-mini-4k-instruct", device="cpu"
        )

    def test_run(self, mock_pipeline_tokenizer, chat_messages):
        generator = OpenVINOChatGenerator(model="microsoft/Phi-3-mini-4k-instruct")

        # Use the mocked pipeline from the fixture and simulate warm_up
        generator.pipeline = mock_pipeline_tokenizer

        results = generator.run(messages=chat_messages)

        assert "replies" in results
        assert isinstance(results["replies"][0], ChatMessage)
        chat_message = results["replies"][0]
        assert chat_message.is_from(ChatRole.ASSISTANT)
        assert chat_message.content == "Berlin is cool"
        assert chat_message.content == "Berlin is cool"
