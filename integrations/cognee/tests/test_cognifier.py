# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

from haystack_integrations.components.connectors.cognee import CogneeCognifier


class TestCogneeCognifier:
    def test_init_defaults(self):
        cognifier = CogneeCognifier()
        assert cognifier.dataset_name is None

    def test_init_custom_string(self):
        cognifier = CogneeCognifier(dataset_name="my_data")
        assert cognifier.dataset_name == "my_data"

    def test_init_custom_list(self):
        cognifier = CogneeCognifier(dataset_name=["ds1", "ds2"])
        assert cognifier.dataset_name == ["ds1", "ds2"]

    def test_to_dict(self):
        cognifier = CogneeCognifier(dataset_name="test_ds")
        data = cognifier.to_dict()
        assert data["type"] == "haystack_integrations.components.connectors.cognee.cognifier.CogneeCognifier"
        assert data["init_parameters"]["dataset_name"] == "test_ds"

    def test_to_dict_defaults(self):
        cognifier = CogneeCognifier()
        data = cognifier.to_dict()
        assert data["init_parameters"]["dataset_name"] is None

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.connectors.cognee.cognifier.CogneeCognifier",
            "init_parameters": {"dataset_name": "restored"},
        }
        cognifier = CogneeCognifier.from_dict(data)
        assert cognifier.dataset_name == "restored"

    @patch("haystack_integrations.components.connectors.cognee.cognifier.cognee")
    def test_run_no_dataset(self, mock_cognee):
        mock_cognee.cognify = AsyncMock()

        cognifier = CogneeCognifier()
        result = cognifier.run()

        assert result == {"cognified": True}
        mock_cognee.cognify.assert_awaited_once_with()

    @patch("haystack_integrations.components.connectors.cognee.cognifier.cognee")
    def test_run_with_string_dataset(self, mock_cognee):
        mock_cognee.cognify = AsyncMock()

        cognifier = CogneeCognifier(dataset_name="my_data")
        result = cognifier.run()

        assert result == {"cognified": True}
        mock_cognee.cognify.assert_awaited_once_with(datasets=["my_data"])

    @patch("haystack_integrations.components.connectors.cognee.cognifier.cognee")
    def test_run_with_list_dataset(self, mock_cognee):
        mock_cognee.cognify = AsyncMock()

        cognifier = CogneeCognifier(dataset_name=["ds1", "ds2"])
        result = cognifier.run()

        assert result == {"cognified": True}
        mock_cognee.cognify.assert_awaited_once_with(datasets=["ds1", "ds2"])

    @patch("haystack_integrations.components.connectors.cognee.cognifier.cognee")
    def test_run_with_documents_written_input(self, mock_cognee):
        mock_cognee.cognify = AsyncMock()

        cognifier = CogneeCognifier(dataset_name="my_data")
        result = cognifier.run(documents_written=5)

        assert result == {"cognified": True}
        mock_cognee.cognify.assert_awaited_once_with(datasets=["my_data"])
