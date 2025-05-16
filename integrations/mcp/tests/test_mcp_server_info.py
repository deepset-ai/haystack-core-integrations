import pytest

from haystack_integrations.tools.mcp import (
    SSEServerInfo,
    StdioServerInfo,
)


class TestMCPServerInfo:
    """Unit tests for MCPServerInfo classes."""

    def test_http_server_info_serde(self):
        """Test serialization/deserialization of SSEServerInfo."""
        server_info = SSEServerInfo(base_url="http://example.com", token="test-token", timeout=45)

        # Test to_dict
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.SSEServerInfo"
        assert info_dict["base_url"] == "http://example.com"
        assert info_dict["token"] == "test-token"
        assert info_dict["timeout"] == 45

        # Test from_dict
        new_info = SSEServerInfo.from_dict(info_dict)
        assert new_info.base_url == "http://example.com"
        assert new_info.token == "test-token"
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

    def test_stdio_server_info_serde(self):
        """Test serialization/deserialization of StdioServerInfo."""
        server_info = StdioServerInfo(command="python", args=["-m", "mcp_server_time"], env={"TEST_ENV": "value"})

        # Test to_dict
        info_dict = server_info.to_dict()
        assert info_dict["type"] == "haystack_integrations.tools.mcp.mcp_tool.StdioServerInfo"
        assert info_dict["command"] == "python"
        assert info_dict["args"] == ["-m", "mcp_server_time"]
        assert info_dict["env"] == {"TEST_ENV": "value"}

        # Test from_dict
        new_info = StdioServerInfo.from_dict(info_dict)
        assert new_info.command == "python"
        assert new_info.args == ["-m", "mcp_server_time"]
        assert new_info.env == {"TEST_ENV": "value"}

    def test_create_client(self):
        """Test client creation from server info."""
        http_info = SSEServerInfo(base_url="http://example.com")
        stdio_info = StdioServerInfo(command="python")

        http_client = http_info.create_client()
        stdio_client = stdio_info.create_client()

        assert http_client.url == "http://example.com/sse"
        assert stdio_client.command == "python"
