# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret

from haystack_integrations.tools.plivo.call_tool import MakeCallTool
from haystack_integrations.tools.plivo.client import PlivoClient
from haystack_integrations.tools.plivo.lookup_tool import LookupNumberTool
from haystack_integrations.tools.plivo.sms_tool import SendSMSTool
from haystack_integrations.tools.plivo.toolset import PlivoToolset
from haystack_integrations.tools.plivo.verify_tool import SendVerificationTool, ValidateVerificationTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**kwargs) -> PlivoClient:
    defaults = {
        "auth_id": Secret.from_token("MA-test-auth-id"),
        "auth_token": Secret.from_token("test-auth-token"),
    }
    defaults.update(kwargs)
    return PlivoClient(**defaults)


def _client_with_mock(**kwargs) -> tuple[PlivoClient, MagicMock]:
    client = _make_client(**kwargs)
    rest = MagicMock()
    client._client = rest
    return client, rest


# ---------------------------------------------------------------------------
# PlivoClient
# ---------------------------------------------------------------------------


class TestPlivoClientInit:
    def test_defaults(self):
        client = _make_client()
        assert client.sender is None
        assert client._client is None

    def test_custom_sender(self):
        client = _make_client(sender="+14150000000")
        assert client.sender == "+14150000000"

    def test_require_sender_raises_when_missing(self):
        client = _make_client()
        with pytest.raises(RuntimeError, match="No sender configured"):
            client._require_sender()

    def test_require_sender_returns_configured(self):
        client = _make_client(sender="+14150000000")
        assert client._require_sender() == "+14150000000"


class TestPlivoClientWarmUp:
    @patch("haystack_integrations.tools.plivo.client.plivo_import")
    @patch("haystack_integrations.tools.plivo.client.plivo")
    def test_warm_up_builds_rest_client(self, mock_plivo, mock_import):
        mock_import.check.return_value = None
        client = _make_client(sender="+14150000000")
        client.warm_up()
        mock_plivo.RestClient.assert_called_once_with(auth_id="MA-test-auth-id", auth_token="test-auth-token")
        assert client._client is mock_plivo.RestClient.return_value

    @patch("haystack_integrations.tools.plivo.client.plivo_import")
    @patch("haystack_integrations.tools.plivo.client.plivo")
    def test_warm_up_is_idempotent(self, mock_plivo, mock_import):
        mock_import.check.return_value = None
        client = _make_client()
        client.warm_up()
        client.warm_up()
        mock_plivo.RestClient.assert_called_once()

    @patch("haystack_integrations.tools.plivo.client.plivo_import")
    @patch("haystack_integrations.tools.plivo.client.plivo")
    def test_warm_up_raises_on_error(self, mock_plivo, mock_import):
        mock_import.check.return_value = None
        mock_plivo.RestClient.side_effect = Exception("bad credentials")
        client = _make_client()
        with pytest.raises(RuntimeError, match="Failed to initialize Plivo client"):
            client.warm_up()


class TestPlivoClientSerialisation:
    def _env_client(self, **kwargs) -> PlivoClient:
        defaults = {
            "auth_id": Secret.from_env_var("PLIVO_AUTH_ID"),
            "auth_token": Secret.from_env_var("PLIVO_AUTH_TOKEN"),
        }
        defaults.update(kwargs)
        return PlivoClient(**defaults)

    def test_to_dict_keys(self):
        data = self._env_client(sender="+14150000000").to_dict()
        assert "PlivoClient" in data["type"]
        assert data["data"]["sender"] == "+14150000000"
        assert "auth_id" in data["data"]
        assert "auth_token" in data["data"]

    def test_to_dict_does_not_leak_client(self):
        client = self._env_client()
        client._client = MagicMock()
        data = client.to_dict()
        assert "_client" not in data["data"]

    def test_round_trip(self):
        original = self._env_client(sender="+14151112222")
        restored = PlivoClient.from_dict(original.to_dict())
        assert restored.sender == "+14151112222"
        assert restored._client is None
        assert restored.auth_id == original.auth_id


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_send_sms_schema(self):
        tool = SendSMSTool(client=_make_client())
        assert tool.name == "send_sms"
        assert set(tool.parameters["required"]) == {"to", "text"}
        assert tool._plivo_client is not None

    def test_send_verification_schema(self):
        tool = SendVerificationTool(client=_make_client())
        assert tool.name == "send_verification_code"
        assert tool.parameters["required"] == ["recipient"]
        assert tool.parameters["properties"]["channel"]["enum"] == ["sms", "voice"]

    def test_validate_verification_schema(self):
        tool = ValidateVerificationTool(client=_make_client())
        assert tool.name == "validate_verification_code"
        assert set(tool.parameters["required"]) == {"session_uuid", "otp"}

    def test_lookup_schema(self):
        tool = LookupNumberTool(client=_make_client())
        assert tool.name == "lookup_number"
        assert tool.parameters["required"] == ["number"]

    def test_make_call_schema(self):
        tool = MakeCallTool(client=_make_client())
        assert tool.name == "make_call"
        assert set(tool.parameters["required"]) == {"to", "answer_url"}


# ---------------------------------------------------------------------------
# SendSMSTool
# ---------------------------------------------------------------------------


class TestSendSMSTool:
    def test_sends_from_configured_sender(self):
        client, rest = _client_with_mock(sender="+14150000000")
        rest.messages.create.return_value = MagicMock(message="message(s) queued", message_uuid=["uuid-1"])
        tool = SendSMSTool(client=client)

        result = tool.invoke(to="+14151234567", text="hello")

        rest.messages.create.assert_called_once_with(src="+14150000000", dst="+14151234567", text="hello")
        assert "uuid-1" in result

    def test_missing_sender_raises(self):
        client, _ = _client_with_mock()
        tool = SendSMSTool(client=client)
        with pytest.raises(ToolInvocationError, match="No sender configured"):
            tool.invoke(to="+14151234567", text="hi")

    def test_wraps_sdk_error(self):
        client, rest = _client_with_mock(sender="+14150000000")
        rest.messages.create.side_effect = Exception("400 invalid number")
        tool = SendSMSTool(client=client)
        with pytest.raises(ToolInvocationError, match="Failed to send SMS"):
            tool.invoke(to="bad", text="hi")


# ---------------------------------------------------------------------------
# Verify tools
# ---------------------------------------------------------------------------


class TestSendVerificationTool:
    def test_sends_and_returns_session_uuid(self):
        client, rest = _client_with_mock()
        rest.verify_session.create.return_value = MagicMock(session_uuid="sess-123", message="message sent")
        tool = SendVerificationTool(client=client)

        result = tool.invoke(recipient="+14151234567")

        rest.verify_session.create.assert_called_once_with(recipient="+14151234567", channel="sms")
        assert "sess-123" in result

    def test_voice_channel(self):
        client, rest = _client_with_mock()
        rest.verify_session.create.return_value = MagicMock(session_uuid="sess-9", message="ok")
        tool = SendVerificationTool(client=client)
        tool.invoke(recipient="+14151234567", channel="voice")
        rest.verify_session.create.assert_called_once_with(recipient="+14151234567", channel="voice")

    def test_wraps_sdk_error(self):
        client, rest = _client_with_mock()
        rest.verify_session.create.side_effect = Exception("boom")
        tool = SendVerificationTool(client=client)
        with pytest.raises(ToolInvocationError, match="Failed to start verification"):
            tool.invoke(recipient="+14151234567")


class TestValidateVerificationTool:
    def test_verified_only_on_success_message(self):
        client, rest = _client_with_mock()
        rest.verify_session.validate.return_value = MagicMock(message="session validated successfully")
        tool = ValidateVerificationTool(client=client)

        result = tool.invoke(session_uuid="sess-1", otp="123456")

        rest.verify_session.validate.assert_called_once_with("sess-1", otp="123456")
        assert result.startswith("verified")

    def test_non_success_message_is_not_verified(self):
        client, rest = _client_with_mock()
        rest.verify_session.validate.return_value = MagicMock(message="something else")
        tool = ValidateVerificationTool(client=client)

        result = tool.invoke(session_uuid="sess-1", otp="000000")

        assert result.startswith("not verified")

    def test_missing_message_is_not_verified(self):
        client, rest = _client_with_mock()
        response = MagicMock()
        del response.message
        rest.verify_session.validate.return_value = response
        tool = ValidateVerificationTool(client=client)

        result = tool.invoke(session_uuid="sess-1", otp="000000")

        assert result.startswith("not verified")

    def test_sdk_error_is_not_verified_and_does_not_raise(self):
        client, rest = _client_with_mock()
        rest.verify_session.validate.side_effect = Exception("400 incorrect code")
        tool = ValidateVerificationTool(client=client)

        result = tool.invoke(session_uuid="sess-1", otp="999999")

        assert result.startswith("not verified")


# ---------------------------------------------------------------------------
# LookupNumberTool
# ---------------------------------------------------------------------------


class TestLookupNumberTool:
    def test_returns_carrier_summary(self):
        client, rest = _client_with_mock()
        rest.lookup.get.return_value = MagicMock(
            phone_number="+14151234567",
            country=MagicMock(name="United States"),
            carrier=MagicMock(name="Verizon", type="mobile", ported="false"),
        )
        tool = LookupNumberTool(client=client)

        result = tool.invoke(number="+14151234567")

        rest.lookup.get.assert_called_once_with("+14151234567")
        assert "mobile" in result
        assert "+14151234567" in result

    def test_wraps_sdk_error(self):
        client, rest = _client_with_mock()
        rest.lookup.get.side_effect = Exception("404")
        tool = LookupNumberTool(client=client)
        with pytest.raises(ToolInvocationError, match="Failed to look up number"):
            tool.invoke(number="bad")


# ---------------------------------------------------------------------------
# MakeCallTool
# ---------------------------------------------------------------------------


class TestMakeCallTool:
    def test_places_call_from_sender(self):
        client, rest = _client_with_mock(sender="+14150000000")
        rest.calls.create.return_value = MagicMock(message="call fired", request_uuid=["req-1"])
        tool = MakeCallTool(client=client)

        result = tool.invoke(to="+14151234567", answer_url="https://example.com/answer")

        rest.calls.create.assert_called_once_with(
            from_="+14150000000", to_="+14151234567", answer_url="https://example.com/answer"
        )
        assert "req-1" in result

    def test_missing_sender_raises(self):
        client, _ = _client_with_mock()
        tool = MakeCallTool(client=client)
        with pytest.raises(ToolInvocationError, match="No sender configured"):
            tool.invoke(to="+14151234567", answer_url="https://example.com/answer")

    def test_wraps_sdk_error(self):
        client, rest = _client_with_mock(sender="+14150000000")
        rest.calls.create.side_effect = Exception("bad url")
        tool = MakeCallTool(client=client)
        with pytest.raises(ToolInvocationError, match="Failed to place call"):
            tool.invoke(to="+14151234567", answer_url="notaurl")


# ---------------------------------------------------------------------------
# Serialization of tools
# ---------------------------------------------------------------------------


class TestToolSerialisation:
    def _env_tool(self, tool_cls):
        client = PlivoClient(
            auth_id=Secret.from_env_var("PLIVO_AUTH_ID"),
            auth_token=Secret.from_env_var("PLIVO_AUTH_TOKEN"),
            sender="+14150000000",
        )
        return tool_cls(client=client)

    @pytest.mark.parametrize(
        "tool_cls",
        [SendSMSTool, SendVerificationTool, ValidateVerificationTool, LookupNumberTool, MakeCallTool],
    )
    def test_round_trip(self, tool_cls):
        tool = self._env_tool(tool_cls)
        restored = tool_cls.from_dict(tool.to_dict())
        assert restored.name == tool.name
        assert restored._plivo_client.sender == "+14150000000"


# ---------------------------------------------------------------------------
# PlivoToolset
# ---------------------------------------------------------------------------


class TestPlivoToolset:
    def test_contains_five_tools(self):
        ts = PlivoToolset(
            auth_id=Secret.from_token("MA-x"),
            auth_token=Secret.from_token("tok"),
            sender="+14150000000",
        )
        assert len(ts) == 5
        names = {t.name for t in ts}
        assert names == {
            "send_sms",
            "send_verification_code",
            "validate_verification_code",
            "lookup_number",
            "make_call",
        }

    def test_tools_share_one_client(self):
        ts = PlivoToolset(auth_id=Secret.from_token("MA-x"), auth_token=Secret.from_token("tok"))
        assert all(t._plivo_client is ts.client for t in ts)

    def test_round_trip(self):
        ts = PlivoToolset(
            auth_id=Secret.from_env_var("PLIVO_AUTH_ID"),
            auth_token=Secret.from_env_var("PLIVO_AUTH_TOKEN"),
            sender="+14150000000",
        )
        restored = PlivoToolset.from_dict(ts.to_dict())
        assert restored.client.sender == "+14150000000"
        assert len(restored) == 5
