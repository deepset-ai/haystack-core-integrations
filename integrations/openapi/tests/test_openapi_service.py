# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Pipeline
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.byte_stream import ByteStream
from openapi3 import OpenAPI
from openapi3.errors import UnexpectedResponseError

from haystack_integrations.components.connectors.openapi import OpenAPIServiceConnector
from haystack_integrations.components.connectors.openapi.openapi_service import patch_request
from haystack_integrations.components.converters.openapi import OpenAPIServiceToFunctions


@pytest.fixture
def openapi_service_mock():
    return MagicMock(spec=OpenAPI)


def _make_response(status_code=200, content_type="application/json", json_data=None, text=""):
    response = Mock()
    response.status_code = status_code
    response.headers = {"Content-Type": content_type}
    response.json.return_value = json_data
    response.text = text
    return response


def _make_operation(responses, request_body=None, security=None):
    operation = Mock()
    operation.path = ["/resource", "get"]
    operation.responses = responses
    operation.requestBody = request_body
    operation.security = security
    operation.operationId = "resource"
    return operation


def _expected_response(content):
    expected = Mock()
    expected.content = content
    return expected


def _json_media_content(model_value):
    media = Mock()
    media.schema.model.return_value = model_value
    return {"application/json": media}


def _service_without_operation():
    service = Mock()
    service.call_missing = "not callable"
    service.info.title = "Test Service"
    return service


def _unsatisfiable_security():
    security = MagicMock()
    security.__iter__.return_value = iter([])
    security.keys.return_value = ["apiKey"]
    return security


class TestOpenAPIServiceConnector:
    @pytest.fixture
    def connector(self):
        return OpenAPIServiceConnector()

    def test_run_without_tool_calls(self, connector):
        message = ChatMessage.from_assistant(text="Just a regular message")
        with pytest.raises(ValueError, match="has no tool calls"):
            connector.run(messages=[message], service_openapi_spec={})

    def test_run_with_non_assistant_message(self, connector):
        message = ChatMessage.from_user(text="User message")
        with pytest.raises(ValueError, match="is not from the assistant"):
            connector.run(messages=[message], service_openapi_spec={})

    def test_authenticate_service_missing_authentication_token(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict

        with pytest.raises(ValueError, match="requires authentication but no credentials were provided"):
            connector._authenticate_service(openapi_service_mock)

    def test_authenticate_service_having_authentication_token(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {
            "apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}
        }
        connector._authenticate_service(openapi_service_mock, "some_fake_token")
        openapi_service_mock.authenticate.assert_called_once_with("apiKey", "some_fake_token")

    def test_authenticate_service_having_authentication_dict(self, connector, openapi_service_mock):
        security_schemes_dict = {
            "components": {"securitySchemes": {"apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}}}
        }
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {
            "apiKey": {"in": "header", "name": "x-api-key", "type": "apiKey"}
        }
        connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})
        openapi_service_mock.authenticate.assert_called_once_with("apiKey", "some_fake_token")

    def test_authenticate_service_having_unsupported_auth(self, connector, openapi_service_mock):
        security_schemes_dict = {"components": {"securitySchemes": {"oauth2": {"type": "oauth2"}}}}
        openapi_service_mock.raw_element = security_schemes_dict
        openapi_service_mock.components.securitySchemes.raw_element = {"oauth2": {"type": "oauth2"}}
        with pytest.raises(ValueError, match="Check the service configuration and credentials"):
            connector._authenticate_service(openapi_service_mock, {"apiKey": "some_fake_token"})

    @patch("haystack_integrations.components.connectors.openapi.openapi_service.OpenAPI")
    def test_run_with_parameters(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="compare_branches",
            arguments={"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"},
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_compare_branches = Mock(return_value={"status": "success"})
        call_compare_branches.operation.__self__ = Mock()
        call_compare_branches.operation.__self__.raw_element = {
            "parameters": [{"name": "basehead"}, {"name": "owner"}, {"name": "repo"}]
        }
        mock_service = Mock(call_compare_branches=call_compare_branches, raw_element={})
        openapi_mock.return_value = mock_service

        result = connector.run(messages=[message], service_openapi_spec={})

        # Verify the service call
        mock_service.call_compare_branches.assert_called_once_with(
            parameters={"basehead": "main...some_branch", "owner": "deepset-ai", "repo": "haystack"}, raw_response=True
        )
        assert json.loads(result["service_response"][0].text) == {"status": "success"}

    @patch("haystack_integrations.components.connectors.openapi.openapi_service.OpenAPI")
    def test_run_with_request_body(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(tool_name="greet", arguments={"message": "Hello", "name": "John"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_greet = Mock(return_value="Hello, John")
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name"}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }
        mock_service = Mock(call_greet=call_greet, raw_element={})
        openapi_mock.return_value = mock_service

        result = connector.run(messages=[message], service_openapi_spec={})

        # Verify the service call
        mock_service.call_greet.assert_called_once_with(
            parameters={"name": "John"}, data={"message": "Hello"}, raw_response=True
        )
        assert json.loads(result["service_response"][0].text) == "Hello, John"

    @patch("haystack_integrations.components.connectors.openapi.openapi_service.OpenAPI")
    def test_run_with_missing_required_parameter(self, openapi_mock):
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="greet",
            arguments={"message": "Hello"},  # missing required 'name' parameter
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_greet = Mock()
        call_greet.operation.__self__ = Mock()
        call_greet.operation.__self__.raw_element = {
            "parameters": [{"name": "name", "required": True}],
            "requestBody": {
                "content": {"application/json": {"schema": {"properties": {"message": {"type": "string"}}}}}
            },
        }
        mock_service = Mock(call_greet=call_greet, raw_element={})
        openapi_mock.return_value = mock_service

        with pytest.raises(ValueError, match="Missing parameter: 'name' required for the 'greet' operation"):
            connector.run(messages=[message], service_openapi_spec={})

    @patch("haystack_integrations.components.connectors.openapi.openapi_service.OpenAPI")
    def test_run_with_missing_required_parameters_in_request_body(self, openapi_mock):
        """
        Test that the connector raises a ValueError when the request body is missing required parameters.
        """
        connector = OpenAPIServiceConnector()
        tool_call = ToolCall(
            tool_name="post_message",
            arguments={"recipient": "John"},  # only providing URL parameter, no request body data
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call])

        # Mock the OpenAPI service
        call_post_message = Mock()
        call_post_message.operation.__self__ = Mock()
        call_post_message.operation.__self__.raw_element = {
            "parameters": [{"name": "recipient"}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "required": ["message"],  # Mark message as required in schema
                            "properties": {"message": {"type": "string"}},
                        }
                    }
                },
            },
        }
        mock_service = Mock(call_post_message=call_post_message, raw_element={})
        openapi_mock.return_value = mock_service

        with pytest.raises(
            ValueError, match="Missing requestBody parameter: 'message' required for the 'post_message' operation"
        ):
            connector.run(messages=[message], service_openapi_spec={})

        # Verify that the service was never called since validation failed
        call_post_message.assert_not_called()

    def test_serialization(self):
        for test_val in ("myvalue", True, None):
            connector = OpenAPIServiceConnector(test_val)
            serialized = connector.to_dict()
            assert serialized["init_parameters"]["ssl_verify"] == test_val
            deserialized = OpenAPIServiceConnector.from_dict(serialized)
            assert deserialized.ssl_verify == test_val

    def test_serde_in_pipeline(self):
        """
        Test serialization/deserialization of OpenAPIServiceConnector in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        connector = OpenAPIServiceConnector(ssl_verify=True)

        pipeline = Pipeline()
        pipeline.add_component("connector", connector)

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "connector": {
                    "type": "haystack_integrations.components.connectors.openapi.openapi_service.OpenAPIServiceConnector",  # noqa: E501
                    "init_parameters": {"ssl_verify": True},
                }
            },
            "connections": [],
        }

        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    @pytest.mark.parametrize(
        "service, descriptor, exception, match",
        [
            (Mock(), {"name": "", "arguments": {"a": 1}}, ValueError, "Invalid function calling descriptor"),
            (Mock(), {"name": "search", "arguments": {}}, ValueError, "Invalid function calling descriptor"),
            (
                _service_without_operation(),
                {"name": "missing", "arguments": {"a": 1}},
                TypeError,
                "not found in OpenAPI specification",
            ),
        ],
    )
    def test_invoke_method_raises_on_invalid_input(self, connector, service, descriptor, exception, match):
        with pytest.raises(exception, match=match):
            connector._invoke_method(service, descriptor)

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run_live(self):
        # An OutputAdapter filter we'll use to setup function calling
        def prepare_fc_params(openai_functions_schema: dict[str, Any]) -> dict[str, Any]:
            return {
                "tools": [{"type": "function", "function": openai_functions_schema}],
                "tool_choice": {"type": "function", "function": {"name": openai_functions_schema["name"]}},
            }

        # Open-Meteo is a free, keyless weather API, so this e2e test needs no service credentials.
        open_meteo_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Open-Meteo Historical Weather API", "version": "1.0.0"},
            "servers": [{"url": "https://archive-api.open-meteo.com"}],
            "paths": {
                "/v1/archive": {
                    "get": {
                        "operationId": "get_archive",
                        "description": "Get historical daily weather data for a location and date range.",
                        "parameters": [
                            {
                                "name": "latitude",
                                "in": "query",
                                "required": True,
                                "description": "Latitude of the location",
                                "schema": {"type": "number"},
                            },
                            {
                                "name": "longitude",
                                "in": "query",
                                "required": True,
                                "description": "Longitude of the location",
                                "schema": {"type": "number"},
                            },
                            {
                                "name": "start_date",
                                "in": "query",
                                "required": True,
                                "description": "Start date in YYYY-MM-DD format",
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "end_date",
                                "in": "query",
                                "required": True,
                                "description": "End date in YYYY-MM-DD format",
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "daily",
                                "in": "query",
                                "required": True,
                                "description": "Comma-separated daily weather variables, e.g. temperature_2m_max",
                                "schema": {"type": "string"},
                            },
                        ],
                        "responses": {
                            "200": {
                                "description": "Historical weather data",
                                "content": {"application/json": {"schema": {"type": "object"}}},
                            }
                        },
                    }
                }
            },
        }

        pipe = Pipeline()
        pipe.add_component("spec_to_functions", OpenAPIServiceToFunctions())
        pipe.add_component("functions_llm", OpenAIChatGenerator(model="gpt-4.1-nano"))

        pipe.add_component("openapi_container", OpenAPIServiceConnector())
        pipe.add_component(
            "prepare_fc_adapter",
            OutputAdapter("{{functions[0] | prepare_fc}}", dict[str, Any], {"prepare_fc": prepare_fc_params}),
        )
        pipe.add_component("openapi_spec_adapter", OutputAdapter("{{specs[0]}}", dict[str, Any], unsafe=True))
        pipe.add_component(
            "final_prompt_adapter",
            OutputAdapter("{{system_message + service_response}}", list[ChatMessage], unsafe=True),
        )
        pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4.1-nano"))

        pipe.connect("spec_to_functions.functions", "prepare_fc_adapter.functions")
        pipe.connect("spec_to_functions.openapi_specs", "openapi_spec_adapter.specs")
        pipe.connect("prepare_fc_adapter", "functions_llm.generation_kwargs")
        pipe.connect("functions_llm.replies", "openapi_container.messages")
        pipe.connect("openapi_spec_adapter", "openapi_container.service_openapi_spec")
        pipe.connect("openapi_container.service_response", "final_prompt_adapter.service_response")
        pipe.connect("final_prompt_adapter", "llm.messages")

        system_prompt = "You are a helpful assistant. Use the provided weather data to answer the user's question."
        query = (
            "What was the daily maximum temperature (temperature_2m_max) in Berlin "
            "(latitude 52.52, longitude 13.41) from 2024-01-01 to 2024-01-07?"
        )

        result = pipe.run(
            data={
                "functions_llm": {
                    "messages": [ChatMessage.from_system("Only do tool/function calling"), ChatMessage.from_user(query)]
                },
                "spec_to_functions": {"sources": [ByteStream.from_string(json.dumps(open_meteo_spec))]},
                "final_prompt_adapter": {"system_message": [ChatMessage.from_system(system_prompt)]},
            },
            include_outputs_from={"openapi_container"},
        )

        # the final LLM produced an answer
        assert isinstance(result["llm"]["replies"][0], ChatMessage)
        assert result["llm"]["replies"][0].text

        # the OpenAPIServiceConnector actually invoked Open-Meteo with the LLM-extracted parameters
        weather_data = json.loads(result["openapi_container"]["service_response"][0].text)
        assert weather_data["latitude"] == pytest.approx(52.52, abs=0.5)
        assert weather_data["longitude"] == pytest.approx(13.41, abs=0.5)


class TestPatchRequest:
    @pytest.mark.parametrize(
        "content, response, raw_response, expected",
        [
            ({"application/json": Mock()}, _make_response(json_data={"ok": True}), True, {"ok": True}),
            ({"application/json": Mock()}, _make_response(content_type="text/plain", text="hello"), True, "hello"),
            (None, _make_response(), False, None),
            (_json_media_content("validated-model"), _make_response(json_data={"x": 1}), False, "validated-model"),
        ],
    )
    def test_patch_request_returns_response(self, content, response, raw_response, expected):
        operation = _make_operation(responses={"200": _expected_response(content)})
        session = Mock()
        session.send.return_value = response
        assert patch_request(operation, "https://example.com", raw_response=raw_response, session=session) == expected

    @pytest.mark.parametrize(
        "operation, response, call_kwargs, exception, match",
        [
            (
                _make_operation(responses={"200": Mock()}, request_body=Mock(required=True)),
                _make_response(),
                {},
                ValueError,
                "Request Body is required",
            ),
            (_make_operation(responses={}), _make_response(status_code=404), {}, UnexpectedResponseError, None),
            (
                _make_operation(responses={"200": _expected_response({"application/json": Mock()})}),
                _make_response(content_type="text/csv"),
                {},
                RuntimeError,
                "Unexpected Content-Type",
            ),
            (
                _make_operation(responses={"200": _expected_response({"text/html": Mock()})}),
                _make_response(content_type="text/html"),
                {},
                NotImplementedError,
                "Only application/json",
            ),
            (
                _make_operation(responses={}, security=_unsatisfiable_security()),
                _make_response(),
                {"security": {"apiKey": "token"}},
                ValueError,
                "No security requirement satisfied",
            ),
        ],
    )
    def test_patch_request_raises(self, operation, response, call_kwargs, exception, match):
        session = Mock()
        session.send.return_value = response
        with pytest.raises(exception, match=match):
            patch_request(operation, "https://example.com", session=session, **call_kwargs)
