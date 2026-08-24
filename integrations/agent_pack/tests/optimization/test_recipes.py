import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.tools import tool

from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    ModelAsset,
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    ToolAsset,
    ToolSelectionRecipe,
)
from haystack_integrations.agent_pack.optimization.recipes import recipe_fingerprint, recipe_from_dict


@tool
def search_documents(query: str) -> str:
    """Search documents."""
    return query


@tool
def calculator(expression: str) -> str:
    """Calculate an expression."""
    return expression


def assets():
    return ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="provider", deployment="remote"),
            ModelAsset(model_id="cheaper", provider="provider", deployment="local"),
        ],
        tools=[
            ToolAsset(name="search_documents"),
            ToolAsset(name="calculator"),
        ],
    )


def reference_agent(**kwargs):
    defaults = {
        "chat_generator": MockChatGenerator(model="reference"),
        "tools": [search_documents, calculator],
        "system_prompt": "reference prompt",
    }
    return Agent(**{**defaults, **kwargs})


def test_model_prompt_and_tool_recipes_clone_without_mutating_reference():
    reference = reference_agent()

    substituted = ModelSubstitutionRecipe(model_id="cheaper").materialize(reference, assets())
    prompted = PromptAndGenerationRecipe(system_prompt="candidate prompt").materialize(reference, assets())
    selected = ToolSelectionRecipe(tool_names=("search_documents",)).materialize(reference, assets())

    assert substituted.chat_generator.model == "cheaper"
    assert substituted.chat_generator is not reference.chat_generator
    assert prompted.system_prompt == "candidate prompt"
    assert [t.name for t in selected.tools] == ["search_documents"]
    assert reference.chat_generator.model == "reference"
    assert reference.system_prompt == "reference prompt"
    assert [t.name for t in reference.tools] == ["search_documents", "calculator"]


class ConfigurableGenerator:
    """A minimal stand-in for a chat generator that accepts generation parameters."""

    def __init__(self, *, model: str = "reference", generation_kwargs: dict | None = None) -> None:
        self.model = model
        self.generation_kwargs = generation_kwargs or {"temperature": 1.0}

    def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - signature only, the reply is fixed
        return {"replies": []}


def test_generation_kwargs_changes_do_not_touch_the_reference_generator():
    reference = reference_agent(chat_generator=ConfigurableGenerator())
    candidate = PromptAndGenerationRecipe(generation_kwargs={"temperature": 0}).materialize(reference, assets())

    assert candidate.chat_generator is not reference.chat_generator
    assert candidate.chat_generator.generation_kwargs == {"temperature": 0}
    assert candidate.chat_generator.model == "reference"
    assert reference.chat_generator.generation_kwargs == {"temperature": 1.0}


def test_generation_kwargs_changes_are_rejected_for_generators_that_do_not_accept_them():
    reference = reference_agent()
    with pytest.raises(ValueError, match="does not accept generation_kwargs"):
        PromptAndGenerationRecipe(generation_kwargs={"temperature": 0}).materialize(reference, assets())


def test_candidates_do_not_share_mutable_containers_with_the_reference():
    """A candidate that registers a hook or appends a tool must not change the reference harness."""
    reference = reference_agent(hooks={"after_run": [BackupAnswerHook(chat_generator=MockChatGenerator("backup"))]})

    candidate = ModelSubstitutionRecipe(model_id="cheaper").materialize(reference, assets())
    candidate.hooks["after_run"].append(BackupAnswerHook(chat_generator=MockChatGenerator("extra")))
    candidate.hooks["before_tool"] = []
    candidate.tools.append(calculator)
    candidate.state_schema["injected"] = {"type": str}

    assert candidate.hooks is not reference.hooks
    assert len(reference.hooks["after_run"]) == 1
    assert "before_tool" not in reference.hooks
    assert [t.name for t in reference.tools] == ["search_documents", "calculator"]
    assert "injected" not in reference.state_schema


def test_tool_selection_normalizes_names_and_prunes_unreachable_exit_conditions():
    reference = reference_agent(exit_conditions=["calculator"])
    recipe = ToolSelectionRecipe(tool_names=("search_documents", "search_documents"))

    assert recipe.tool_names == ("search_documents",)
    assert recipe_fingerprint(recipe=recipe) == recipe_fingerprint(
        recipe=ToolSelectionRecipe(tool_names=("search_documents",))
    )

    candidate = recipe.materialize(reference, assets())
    # `calculator` is gone, so keeping it as an exit condition would fail Agent construction.
    assert candidate.exit_conditions == ["text"]
    assert reference.exit_conditions == ["calculator"]


def test_recipe_parser_is_closed_and_fingerprints_are_stable():
    data = {"kind": "prompt_and_generation", "system_prompt": "candidate prompt"}
    first = recipe_from_dict(data=data)
    second = recipe_from_dict(data=data)
    assert recipe_fingerprint(recipe=first) == recipe_fingerprint(recipe=second)
    with pytest.raises(ValueError, match="Unsupported candidate recipe kind"):
        recipe_from_dict(data={"kind": "python", "code": "dangerous()"})
    for removed in ("specialist_delegation", "composite", "registered_structure"):
        with pytest.raises(ValueError, match="Unsupported candidate recipe kind"):
            recipe_from_dict(data={"kind": removed})


def test_unknown_tools_and_empty_changes_are_rejected():
    with pytest.raises(ValueError, match="not configured"):
        ToolSelectionRecipe(tool_names=("unknown",)).materialize(reference_agent(), assets())
    with pytest.raises(ValueError, match="at least one tool name"):
        ToolSelectionRecipe(tool_names=())
    with pytest.raises(ValueError, match="requires at least one change"):
        PromptAndGenerationRecipe()
