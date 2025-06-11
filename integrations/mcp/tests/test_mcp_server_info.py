import pytest
from haystack.utils import Secret

from haystack_integrations.tools.mcp import (
    SSEServerInfo,
    StdioServerInfo,
    StreamableHttpServerInfo,
)


class TestMCPServerInfo:
    """Unit tests for MCPServerInfo classes."""

    def test_http_server_info_serde(self):
        """Test serialization/deserialization of SSEServerInfo with plain string token."""
        server_info = SSEServerInfo(base_url="http://example.com", token="test-token", timeout=45)

        # Test to_dict - all fields are serialized including plain string tokens
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"
        assert info_dict["base_url"] == "http://example.com"
        assert info_dict["token"] == "test-token"  # Plain string tokens are included
        assert info_dict["timeout"] == 45

        # Test from_dict - plain string token is preserved
        new_info = SSEServerInfo.from_dict(info_dict)
        assert new_info.base_url == "http://example.com"
        assert new_info.token == "test-token"  # Token preserved as-is
        assert new_info.timeout == 45

    def test_streamable_http_server_info_serde(self):
        """Test serialization/deserialization of StreamableHttpServerInfo with plain string token."""
        server_info = StreamableHttpServerInfo(url="http://example.com/mcp", token="test-token", timeout=45)

        # Test to_dict - all fields are serialized including plain string tokens
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo"
        assert info_dict["url"] == "http://example.com/mcp"
        assert info_dict["token"] == "test-token"  # Plain string tokens are included
        assert info_dict["timeout"] == 45

        # Test from_dict - plain string token is preserved
        new_info = StreamableHttpServerInfo.from_dict(info_dict)
        assert new_info.url == "http://example.com/mcp"
        assert new_info.token == "test-token"  # Token preserved as-is
        assert new_info.timeout == 45

    def test_url_base_url_validation(self):
        """Test validation of url and base_url parameters."""
        # Test with neither url nor base_url
        with pytest.raises(ValueError, match="Either url or base_url must be provided"):
            SSEServerInfo()

        # Test with both url and base_url
        with pytest.warns(DeprecationWarning, match="base_url is deprecated"):
            SSEServerInfo(url="http://example.com/sse", base_url="http://example.com")

        # Test with only url
        server_info = SSEServerInfo(url="http://example.com/sse")
        assert server_info.url == "http://example.com/sse"
        assert server_info.base_url is None

        # Test with only base_url (deprecated but supported)
        with pytest.warns(DeprecationWarning, match="base_url is deprecated"):
            server_info = SSEServerInfo(base_url="http://example.com")
            assert server_info.base_url == "http://example.com"  # Should preserve original base_url

    def test_streamable_http_url_validation(self):
        """Test URL validation for StreamableHttpServerInfo."""
        # Test with valid URL
        server_info = StreamableHttpServerInfo(url="http://example.com/mcp")
        assert server_info.url == "http://example.com/mcp"

        # Test with invalid URL
        with pytest.raises(ValueError, match="Invalid url:"):
            StreamableHttpServerInfo(url="not-a-url")

    def test_stdio_server_info_serde(self):
        """Test serialization/deserialization of StdioServerInfo with plain string env vars."""
        server_info = StdioServerInfo(command="python", args=["-m", "mcp_server_time"], env={"TEST_ENV": "value"})

        # Test to_dict - all fields are serialized including plain string env vars
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo"
        assert info_dict["command"] == "python"
        assert info_dict["args"] == ["-m", "mcp_server_time"]
        assert info_dict["env"] == {"TEST_ENV": "value"}  # Plain string env vars are included

        # Test from_dict - plain string env vars are preserved
        new_info = StdioServerInfo.from_dict(info_dict)
        assert new_info.command == "python"
        assert new_info.args == ["-m", "mcp_server_time"]
        assert new_info.env == {"TEST_ENV": "value"}  # Env preserved as-is

    def test_create_client(self):
        """Test client creation from server info."""
        http_info = SSEServerInfo(base_url="http://example.com")
        stdio_info = StdioServerInfo(command="python")
        streamable_http_info = StreamableHttpServerInfo(url="http://example.com/mcp")

        http_client = http_info.create_client()
        stdio_client = stdio_info.create_client()
        streamable_http_client = streamable_http_info.create_client()

        assert http_client.url == "http://example.com/sse"
        assert stdio_client.command == "python"
        assert streamable_http_client.url == "http://example.com/mcp"

    def test_sse_server_info_with_secret_token(self, monkeypatch):
        """Test SSEServerInfo serialization/deserialization with Secret token."""
        monkeypatch.setenv("TEST_TOKEN", "secret_token_value")
        secret_token = Secret.from_env_var("TEST_TOKEN")
        server_info = SSEServerInfo(url="http://example.com/sse", token=secret_token, timeout=45)

        # Test to_dict - Secret tokens are serialized
        info_dict = server_info.to_dict()
        info_dict = {
            "type": "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo",
            "url": "http://example.com/sse",
            "base_url": None,
            "token": {"type": "env_var", "env_vars": ["TEST_TOKEN"], "strict": True},
            "timeout": 45,
        }

        # Test from_dict - Secret is properly deserialized
        new_info = SSEServerInfo.from_dict(info_dict)
        assert new_info.url == "http://example.com/sse"
        assert isinstance(new_info.token, Secret)
        assert new_info.token.resolve_value() == "secret_token_value"
        assert new_info.timeout == 45

    def test_streamable_http_server_info_with_secret_token(self, monkeypatch):
        """Test StreamableHttpServerInfo serialization/deserialization with Secret token."""
        monkeypatch.setenv("TEST_TOKEN", "secret_token_value")
        secret_token = Secret.from_env_var("TEST_TOKEN")
        server_info = StreamableHttpServerInfo(url="http://example.com/mcp", token=secret_token, timeout=45)

        # Test to_dict - Secret tokens are serialized
        info_dict = server_info.to_dict()
        assert info_dict == {
            "type": "haystack_integrations.tools.mcp.mcp_tool.StreamableHttpServerInfo",
            "url": "http://example.com/mcp",
            "token": {"type": "env_var", "env_vars": ["TEST_TOKEN"], "strict": True},
            "timeout": 45,
        }

        # Test from_dict - Secret is properly deserialized
        new_info = StreamableHttpServerInfo.from_dict(info_dict)
        assert new_info.url == "http://example.com/mcp"
        assert isinstance(new_info.token, Secret)
        assert new_info.token.resolve_value() == "secret_token_value"
        assert new_info.timeout == 45

    def test_stdio_server_info_with_secret_env_vars(self, monkeypatch):
        """Test StdioServerInfo serialization/deserialization with Secret env vars."""
        monkeypatch.setenv("SECRET_VAR", "secret_var_value")
        secret_env = {
            "PUBLIC_VAR": "public_value",
            "SECRET_VAR": Secret.from_env_var("SECRET_VAR"),
        }
        server_info = StdioServerInfo(command="python", args=["-m", "server"], env=secret_env)

        # Test to_dict - all env vars are serialized, Secrets get proper serialization
        info_dict = server_info.to_dict()
        assert info_dict == {
            "type": "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo",
            "command": "python",
            "args": ["-m", "server"],
            "env": {
                "PUBLIC_VAR": "public_value",
                "SECRET_VAR": {"type": "env_var", "env_vars": ["SECRET_VAR"], "strict": True},
            },
        }

        # Test from_dict - Secret env vars are properly deserialized, plain strings stay as-is
        new_info = StdioServerInfo.from_dict(info_dict)
        assert new_info.command == "python"
        assert new_info.args == ["-m", "server"]
        assert isinstance(new_info.env, dict)
        assert "SECRET_VAR" in new_info.env
        assert isinstance(new_info.env["SECRET_VAR"], Secret)
        assert new_info.env["SECRET_VAR"].resolve_value() == "secret_var_value"
        assert "PUBLIC_VAR" in new_info.env
        assert new_info.env["PUBLIC_VAR"] == "public_value"

    def test_stdio_server_info_with_only_secret_env_vars(self, monkeypatch):
        """Test StdioServerInfo with only Secret env vars (no plain strings)."""
        monkeypatch.setenv("SECRET_VAR1", "secret_var_value")
        monkeypatch.setenv("SECRET_VAR2", "secret_var_value")
        secret_env = {
            "SECRET_VAR1": Secret.from_env_var("SECRET_VAR1"),
            "SECRET_VAR2": Secret.from_env_var("SECRET_VAR2"),
        }
        server_info = StdioServerInfo(command="python", env=secret_env)

        # Test serialization and deserialization
        info_dict = server_info.to_dict()
        assert info_dict == {
            "type": "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo",
            "command": "python",
            "args": None,
            "env": {
                "SECRET_VAR1": {"type": "env_var", "env_vars": ["SECRET_VAR1"], "strict": True},
                "SECRET_VAR2": {"type": "env_var", "env_vars": ["SECRET_VAR2"], "strict": True},
            },
        }
        new_info = StdioServerInfo.from_dict(info_dict)
        assert len(new_info.env) == 2
        assert isinstance(new_info.env["SECRET_VAR1"], Secret)
        assert isinstance(new_info.env["SECRET_VAR2"], Secret)
        assert new_info.env["SECRET_VAR1"].resolve_value() == "secret_var_value"
        assert new_info.env["SECRET_VAR2"].resolve_value() == "secret_var_value"

    def test_stdio_server_info_with_only_plain_env_vars(self):
        """Test StdioServerInfo with only plain string env vars."""
        server_info = StdioServerInfo(
            command="python",
            args=["server.py"],
            env={"PLAIN_VAR": "plain_value", "ANOTHER_VAR": "another_value"},
        )

        serialized = server_info.to_dict()
        assert serialized["env"] == {"PLAIN_VAR": "plain_value", "ANOTHER_VAR": "another_value"}

        deserialized = StdioServerInfo.from_dict(serialized)
        assert deserialized.env == {"PLAIN_VAR": "plain_value", "ANOTHER_VAR": "another_value"}

    def test_sse_server_info_with_token_secret_cannot_serialize(self):
        """Test that SSEServerInfo with Secret.from_token cannot be serialized."""
        server_info = SSEServerInfo(
            url="https://localhost:8000/sse",
            token=Secret.from_token("secret_token_value"),
        )

        # Token-based secrets cannot be serialized
        with pytest.raises(ValueError, match="token"):
            server_info.to_dict()

    def test_secret_deserialization_handles_any_secret_type(self, monkeypatch):
        """Test that our deserialization can handle any Secret type, not just env_vars."""
        monkeypatch.setenv("TEST_VAR", "test_var_value")
        fake_secret_dict = {"type": "env_var", "env_vars": ["TEST_VAR"], "strict": True}  # Valid Secret type

        server_info_dict = {
            "type": "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo",
            "url": "https://localhost:8000/sse",
            "token": fake_secret_dict,
            "timeout": 30,
            "base_url": None,
        }

        # This should work - our condition only checks for "type" field
        deserialized = SSEServerInfo.from_dict(server_info_dict)
        assert isinstance(deserialized.token, Secret)
        assert deserialized.token.resolve_value() == "test_var_value"
        assert deserialized.url == "https://localhost:8000/sse"

    def test_non_secret_dict_with_type_field_not_deserialized(self):
        """Test that dictionaries with non-Secret "type" fields are left as-is."""
        # Create a dict with a "type" field that"s NOT a Secret type
        non_secret_dict = {"type": "some_other_type", "data": "some_value"}  # Not "token" or "env_var"

        server_info_dict = {
            "type": "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo",
            "command": "python",
            "args": ["server.py"],
            "env": {"CONFIG": non_secret_dict},  # This should NOT be treated as a Secret
        }

        # Deserialize - the non-Secret dict should be preserved as-is
        deserialized = StdioServerInfo.from_dict(server_info_dict)
        assert deserialized.env["CONFIG"] == non_secret_dict  # Should be unchanged
        assert not isinstance(deserialized.env["CONFIG"], Secret)  # Should NOT be a Secret
