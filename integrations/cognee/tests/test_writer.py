# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

from haystack import Document

from haystack_integrations.components.writers.cognee import CogneeWriter


class TestCogneeWriter:
    def test_init_defaults(self):
        writer = CogneeWriter()
        assert writer.dataset_name == "haystack"
        assert writer.auto_cognify is True

    def test_init_custom(self):
        writer = CogneeWriter(dataset_name="custom", auto_cognify=False)
        assert writer.dataset_name == "custom"
        assert writer.auto_cognify is False

    def test_to_dict(self):
        writer = CogneeWriter(dataset_name="test_ds", auto_cognify=False)
        data = writer.to_dict()
        assert data["type"] == "haystack_integrations.components.writers.cognee.memory_writer.CogneeWriter"
        assert data["init_parameters"]["dataset_name"] == "test_ds"
        assert data["init_parameters"]["auto_cognify"] is False

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.writers.cognee.memory_writer.CogneeWriter",
            "init_parameters": {"dataset_name": "restored", "auto_cognify": True},
        }
        writer = CogneeWriter.from_dict(data)
        assert writer.dataset_name == "restored"
        assert writer.auto_cognify is True

    @patch("haystack_integrations.components.writers.cognee.memory_writer.cognee")
    def test_run_with_auto_cognify(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        writer = CogneeWriter(dataset_name="test", auto_cognify=True)
        docs = [
            Document(content="First document"),
            Document(content="Second document"),
        ]
        result = writer.run(documents=docs)

        assert result == {"documents_written": 2}
        # Verify batch call: single add() with list of texts
        mock_cognee.add.assert_awaited_once()
        call_args = mock_cognee.add.call_args
        assert call_args[0][0] == ["First document", "Second document"]
        # Verify cognify uses specific dataset
        mock_cognee.cognify.assert_awaited_once()
        cognify_kwargs = mock_cognee.cognify.call_args[1]
        assert cognify_kwargs["datasets"] == ["test"]

    @patch("haystack_integrations.components.writers.cognee.memory_writer.cognee")
    def test_run_without_auto_cognify(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        writer = CogneeWriter(dataset_name="test", auto_cognify=False)
        docs = [Document(content="A document")]
        result = writer.run(documents=docs)

        assert result == {"documents_written": 1}
        mock_cognee.add.assert_awaited_once()
        mock_cognee.cognify.assert_not_awaited()

    @patch("haystack_integrations.components.writers.cognee.memory_writer.cognee")
    def test_run_skips_empty_content(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        writer = CogneeWriter(auto_cognify=True)
        docs = [
            Document(content="Valid document"),
            Document(content=""),
            Document(content=None),
        ]
        result = writer.run(documents=docs)

        assert result == {"documents_written": 1}
        mock_cognee.add.assert_awaited_once()
        call_args = mock_cognee.add.call_args
        assert call_args[0][0] == ["Valid document"]

    @patch("haystack_integrations.components.writers.cognee.memory_writer.cognee")
    def test_run_empty_list(self, mock_cognee):
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock()

        writer = CogneeWriter(auto_cognify=True)
        result = writer.run(documents=[])

        assert result == {"documents_written": 0}
        mock_cognee.add.assert_not_awaited()
        mock_cognee.cognify.assert_not_awaited()
