# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from haystack.tools import ComponentTool
from haystack.utils import Secret

from haystack_integrations.tools.mrscraper import MrScraperToolset
from haystack_integrations.tools.mrscraper.toolset import _result_to_string

from .conftest import FAKE_TOKEN

TOOL_NAMES = [
    "mrscraper_get_account_info",
    "mrscraper_crawl_website_urls",
    "mrscraper_search_google_serp",
    "mrscraper_extract_page_by_prompt",
    "mrscraper_extract_listings",
    "mrscraper_extract_structured_data",
    "mrscraper_fetch_rendered_html",
    "mrscraper_get_results",
    "mrscraper_get_latest_results",
    "mrscraper_get_result_detail",
    "mrscraper_create_prompt_scraper",
    "mrscraper_create_listing_scraper",
    "mrscraper_create_website_crawl_scraper",
    "mrscraper_run_existing_scraper",
    "mrscraper_run_existing_scraper_batch",
]


def toolset(**kwargs):
    return MrScraperToolset(api_key=Secret.from_token(FAKE_TOKEN), **kwargs)


def test_default_toolset_contains_exactly_15_unique_component_tools():
    tools = toolset()
    assert len(tools) == 15
    assert [tool.name for tool in tools] == TOOL_NAMES
    assert len({tool.name for tool in tools}) == 15
    assert all(isinstance(tool, ComponentTool) for tool in tools)
    assert len({id(tool._component) for tool in tools}) == 15


@pytest.mark.parametrize(
    ("groups", "expected_names"),
    [
        (["account"], TOOL_NAMES[:1]),
        (["discovery"], TOOL_NAMES[1:3]),
        (["extraction"], TOOL_NAMES[3:7]),
        (["results"], TOOL_NAMES[7:10]),
        (["scraper_creation"], TOOL_NAMES[10:13]),
        (["scraper_runs"], TOOL_NAMES[13:15]),
        (["discovery", "results"], TOOL_NAMES[1:3] + TOOL_NAMES[7:10]),
        ([], []),
    ],
)
def test_group_selection(groups, expected_names):
    assert [tool.name for tool in toolset(groups=groups)] == expected_names


def test_invalid_and_duplicate_groups_rejected():
    with pytest.raises(ValueError, match="group"):
        toolset(groups=["unknown"])
    with pytest.raises(ValueError, match="duplicate"):
        toolset(groups=["results", "results"])


def test_tool_schemas_are_valid_distinct_and_secret_free():
    tools = toolset()
    serialized_specs = json.dumps([tool.tool_spec for tool in tools])
    assert FAKE_TOKEN not in serialized_specs
    assert "api_key" not in serialized_specs
    assert "connect_timeout" not in serialized_specs
    assert "read_timeout" not in serialized_specs
    for tool in tools:
        assert tool.parameters["type"] == "object"
        assert tool.description
    single_schema = next(tool for tool in tools if tool.name == "mrscraper_run_existing_scraper").parameters
    assert single_schema["properties"]["scraper_type"]["enum"] == ["ai", "manual"]
    branches = {branch["title"]: branch for branch in single_schema["oneOf"]}
    assert set(branches) == {
        "Manual scraper",
        "General AI scraper",
        "Listing AI scraper",
        "Map AI scraper",
    }
    assert "agent_type" not in branches["Manual scraper"]["properties"]
    assert branches["General AI scraper"]["properties"]["agent_type"]["const"] == "general"
    assert "max_pages" in branches["Listing AI scraper"]["properties"]
    assert "max_depth" not in branches["Listing AI scraper"]["properties"]
    assert "max_depth" in branches["Map AI scraper"]["properties"]
    assert "html" not in branches["Map AI scraper"]["properties"]
    batch_schema = next(tool for tool in tools if tool.name == "mrscraper_run_existing_scraper_batch").parameters
    assert batch_schema["properties"]["urls"]["type"] == "array"


def test_toolset_environment_secret_serialization_roundtrip():
    tools = MrScraperToolset(
        api_key=Secret.from_env_var("MRSCRAPER_API_TOKEN"),
        groups=["results"],
        connect_timeout=4,
        read_timeout=20,
    )
    data = tools.to_dict()
    serialized = json.dumps(data)
    assert FAKE_TOKEN not in serialized
    assert data["data"]["api_key"]["env_vars"] == ["MRSCRAPER_API_TOKEN"]
    restored = MrScraperToolset.from_dict(data)
    assert restored.groups == ["results"]
    assert [tool.name for tool in restored] == TOOL_NAMES[7:10]
    assert restored.connect_timeout == 4
    assert restored.read_timeout == 20


def test_toolset_token_secret_is_not_serializable():
    with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
        toolset().to_dict()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<html> exact\n", "<html> exact\n"),
        ({"z": "é", "a": [1, False]}, '{"a":[1,false],"z":"é"}'),
        ([1, "x"], '[1,"x"]'),
        (False, "false"),
        (0, "0"),
        (None, "null"),
    ],
)
def test_agent_result_formatting(value, expected):
    assert _result_to_string(value) == expected


def test_component_tool_invocation_and_output_mapping(http_recorder):
    assert not http_recorder.requests
    account_tool = toolset(groups=["account"])[0]
    result = account_tool.invoke()
    assert result == {"result": {"ok": True}}
    assert account_tool.outputs_to_string["source"] == "result"
    assert account_tool.outputs_to_string["handler"]({"z": 1, "a": 2}) == '{"a":2,"z":1}'
