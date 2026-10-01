import importlib
import json

import pytest
from haystack.core.errors import DeserializationError

from haystack_integrations.agent_pack.optimization.tools import (
    WORKSPACE_TOOLS,
    _documentation_result,
    _make_haystack_documentation_toolset,
    inspect_component,
)


def docs_payload(**body):
    """The shape the documentation server answers with: a payload serialized inside an MCP envelope."""
    return json.dumps({"meta": None, "content": [{"type": "text", "text": json.dumps(body)}]})


class TestDocumentationToolset:
    def test_toolset(self):
        pytest.importorskip("haystack_integrations.tools.mcp")
        docs = _make_haystack_documentation_toolset()
        assert docs.tool_names == ["search_haystack_docs"]
        assert docs.eager_connect is False


class TestDocumentationSearch:
    def test_strips_the_servers_debug_output(self):
        """Measured against the live server, the debug payload is 94% of the answer and says nothing about Haystack."""
        payload = docs_payload(
            documents=[{"content": "LLMRanker reorders documents.", "meta": {"url": "https://docs/llmranker"}}],
            _debug={"pipeline": "x" * 5000},
        )
        result = _documentation_result(payload)
        assert result == "[https://docs/llmranker]\nLLMRanker reorders documents."
        assert "_debug" not in result

    def test_unexpected_answer(self):
        """A server that changes shape must not silently look like an empty search."""
        assert _documentation_result("not json at all") == "not json at all"
        assert _documentation_result(docs_payload(unexpected=1)).startswith('{"meta"')

    def test_no_match(self):
        assert _documentation_result(docs_payload(documents=[])) == "No documentation matched."


class TestInspectComponent:
    def test_reports_the_installed_path(self):
        """The answer carries the string the YAML has to use, not the one that happened to be asked for."""
        answer = inspect_component.function(type_name="haystack.components.rankers.llm_ranker.LLMRanker")
        assert answer["import_path"] == "haystack.components.rankers.llm_ranker.LLMRanker"
        assert "top_k" in answer["constructor"]

    def test_finds_a_class_asked_for_elsewhere(self):
        """
        The reference names its generator in `...chat.openai`, so reaching for the responses one by changing the class
        on the end of that path is the natural mistake. Uncorrected it reaches the YAML and fails deserialization.
        """
        answer = inspect_component.function(
            type_name="haystack.components.generators.chat.openai.OpenAIResponsesChatGenerator"
        )
        assert answer["import_path"] == (
            "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator"
        )

    def test_class_that_is_not_installed(self):
        with pytest.raises(ImportError):
            inspect_component.function(type_name="haystack.components.nonsense.NoSuchComponent")

    def test_cannot_import_outside_the_allowlist(self, monkeypatch):
        """
        Recovering from a wrong path must not become a way to import anything: the search asks for shorter and
        shorter paths, and a module off the allowlist has to be refused before it is executed rather than after.
        """

        def refuse(name, *_args, **_kwargs):
            pytest.fail(f"searching for a class imported {name}")

        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(DeserializationError):
            inspect_component.function(type_name="subprocess.check_output.Popen")

    def test_tool_name(self):
        """The system prompt refers to the tool by this name."""
        assert inspect_component.name == "inspect_component"


class TestWorkspaceTools:
    def test_tool_names(self):
        """The system prompt refers to the tools by these names."""
        assert [tool.name for tool in WORKSPACE_TOOLS] == [
            "read_config",
            "edit_config",
            "validate_config",
            "submit_candidate",
            "restore_candidate",
            "finish",
        ]

    def test_workspace_comes_from_state(self):
        """The optimizer never sees the state, the workspace or a path: the agent injects its live state."""
        for workspace_tool in WORKSPACE_TOOLS:
            assert not {"state", "workspace", "path"} & set(workspace_tool.parameters.get("properties", {}))

    def test_every_parameter_is_described(self):
        """
        `create_tool_from_function` builds the schema from `Annotated` metadata and never reads `:param` lines, so a
        parameter documented only in the docstring reaches the model as a bare string with no explanation of it.
        """
        described = {
            f"{workspace_tool.name}.{name}": specification.get("description")
            for workspace_tool in WORKSPACE_TOOLS
            for name, specification in workspace_tool.parameters.get("properties", {}).items()
        }
        assert described
        assert [parameter for parameter, description in described.items() if not description] == []
        # The constraint that actually fails at runtime has to be in the schema, not only in the error it raises.
        assert "exactly once" in described["edit_config.old"]
