import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.tools import AgentTool, tool

from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    CompositeRecipe,
    ModelAsset,
    ModelSubstitutionRecipe,
    PromptAndGenerationRecipe,
    RegisteredStructuralRecipe,
    SpecialistDelegationRecipe,
    StructuralRecipeRegistry,
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
            ToolAsset(name="retrieval_specialist"),
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


def test_specialist_delegation_uses_agent_tool_and_validates_recursively():
    reference = reference_agent()
    recipe = SpecialistDelegationRecipe(
        name="retrieval_specialist",
        description="Retrieve grounded evidence",
        specialist_tool_names=("search_documents",),
        specialist_system_prompt="Only retrieve evidence.",
        coordinator_tool_names=("calculator",),
        specialist_model_id="cheaper",
    )

    candidate = recipe.materialize(reference, assets())

    assert [t.name for t in candidate.tools] == ["calculator", "retrieval_specialist"]
    specialist_tool = candidate.tools[1]
    assert isinstance(specialist_tool, AgentTool)
    specialist_data = specialist_tool.to_dict()["data"]["agent"]["init_parameters"]
    assert specialist_data["chat_generator"]["init_parameters"]["model"] == "cheaper"
    assert [t["data"]["name"] for t in specialist_data["tools"]] == ["search_documents"]
    assert assets().require_valid_agent(candidate).allowed is True


def test_specialist_is_isolated_and_the_coordinator_is_told_to_delegate():
    reference = reference_agent(exit_conditions=["calculator"], user_prompt="Question: {{question}}")
    candidate = SpecialistDelegationRecipe(
        name="retrieval_specialist",
        description="Retrieve grounded evidence",
        specialist_tool_names=("search_documents",),
        specialist_system_prompt="Only retrieve evidence.",
    ).materialize(reference, assets())

    specialist = candidate.tools[0]._component
    assert specialist.exit_conditions == ["text"]
    assert specialist.user_prompt is None
    assert specialist.system_prompt == "Only retrieve evidence."
    assert candidate.exit_conditions == ["text"]
    assert candidate.system_prompt.startswith("reference prompt")
    assert "retrieval_specialist" in candidate.system_prompt
    assert "search_documents" in candidate.system_prompt


def test_specialist_delegation_accepts_an_explicit_coordinator_prompt():
    candidate = SpecialistDelegationRecipe(
        name="retrieval_specialist",
        description="Retrieve grounded evidence",
        specialist_tool_names=("search_documents",),
        specialist_system_prompt="Only retrieve evidence.",
        coordinator_system_prompt="You coordinate specialists.",
    ).materialize(reference_agent(), assets())
    assert candidate.system_prompt == "You coordinate specialists."


def test_recipe_parser_is_closed_and_fingerprints_are_stable():
    data = {
        "kind": "specialist_delegation",
        "name": "retrieval_specialist",
        "description": "Retrieve evidence",
        "specialist_tool_names": ["search_documents"],
        "specialist_system_prompt": "Retrieve.",
        "coordinator_tool_names": ["calculator"],
        "specialist_model_id": "cheaper",
    }
    first = recipe_from_dict(data=data)
    second = recipe_from_dict(data=data)
    assert recipe_fingerprint(recipe=first) == recipe_fingerprint(recipe=second)
    with pytest.raises(ValueError, match="Unsupported candidate recipe kind"):
        recipe_from_dict(data={"kind": "python", "code": "dangerous()"})


def shorten(agent, _assets, params):
    return agent.clone(max_agent_steps=params["steps"])


def test_composite_and_registered_structural_recipes():
    reference = reference_agent()
    composite = CompositeRecipe(
        recipes=(ModelSubstitutionRecipe(model_id="cheaper"), ToolSelectionRecipe(tool_names=("search_documents",)))
    )
    candidate = composite.materialize(reference, assets())
    assert candidate.chat_generator.model == "cheaper"
    assert [t.name for t in candidate.tools] == ["search_documents"]

    registry = StructuralRecipeRegistry()
    registry.register("short-run", shorten, parameters_schema={"steps": "int"})
    registered = RegisteredStructuralRecipe(name="short-run", parameters={"steps": 3}, registry=registry)
    assert registered.materialize(reference, assets()).max_agent_steps == 3
    restored = recipe_from_dict(data=registered.to_dict(), registry=registry)
    assert restored.materialize(reference, assets()).max_agent_steps == 3
    assert registry.describe() == {"short-run": {"steps": "int"}}


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({}, "requires parameter 'steps'"),
        ({"steps": "three"}, "expects 'steps' to be int"),
        ({"steps": True}, "expects 'steps' to be int"),
        ({"steps": 3, "extra": 1}, "unsupported parameters: extra"),
    ],
)
def test_registered_structural_parameters_are_schema_checked(parameters, message):
    """Proposed parameters must never reach a registered factory unchecked."""
    registry = StructuralRecipeRegistry()
    registry.register("short-run", shorten, parameters_schema={"steps": "int"})
    with pytest.raises(ValueError, match=message):
        RegisteredStructuralRecipe(name="short-run", parameters=parameters, registry=registry).materialize(
            reference_agent(), assets()
        )


def test_optional_schema_parameters_may_be_omitted():
    registry = StructuralRecipeRegistry()
    registry.register(
        "short-run",
        lambda agent, _assets, params: agent.clone(max_agent_steps=params.get("steps", 7)),
        parameters_schema={"steps": "int?"},
    )
    assert (
        RegisteredStructuralRecipe(name="short-run", parameters={}, registry=registry)
        .materialize(reference_agent(), assets())
        .max_agent_steps
        == 7
    )


def test_missing_tools_and_unregistered_structures_are_rejected():
    reference = reference_agent()
    with pytest.raises(ValueError, match="not configured"):
        ToolSelectionRecipe(tool_names=("unknown",)).materialize(reference, assets())
    with pytest.raises(ValueError, match="not registered"):
        RegisteredStructuralRecipe(name="unknown", parameters={}, registry=StructuralRecipeRegistry()).materialize(
            reference, assets()
        )
    with pytest.raises(ValueError, match="at least one tool name"):
        ToolSelectionRecipe(tool_names=())
    with pytest.raises(ValueError, match="at least one recipe"):
        CompositeRecipe(recipes=())
