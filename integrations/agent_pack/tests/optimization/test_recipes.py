import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.tools import AgentTool, tool

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
    recipe_fingerprint,
    recipe_from_dict,
)


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
            ModelAsset("reference", "provider", "remote"),
            ModelAsset("cheaper", "provider", "local", sovereign=True),
        ],
        tools=[ToolAsset("search_documents"), ToolAsset("calculator"), ToolAsset("retrieval_specialist")],
    )


def reference_agent():
    return Agent(
        chat_generator=MockChatGenerator(model="reference"),
        tools=[search_documents, calculator],
        system_prompt="reference prompt",
    )


def test_model_prompt_and_tool_recipes_clone_without_mutating_reference():
    reference = reference_agent()

    substituted = ModelSubstitutionRecipe("cheaper").materialize(reference, assets())
    prompted = PromptAndGenerationRecipe(system_prompt="candidate prompt").materialize(reference, assets())
    selected = ToolSelectionRecipe(("search_documents",)).materialize(reference, assets())

    assert substituted.chat_generator.model == "cheaper"
    assert prompted.system_prompt == "candidate prompt"
    assert [tool.name for tool in selected.tools] == ["search_documents"]
    assert reference.chat_generator.model == "reference"
    assert reference.system_prompt == "reference prompt"
    assert [tool.name for tool in reference.tools] == ["search_documents", "calculator"]


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

    assert [tool.name for tool in candidate.tools] == ["calculator", "retrieval_specialist"]
    specialist_tool = candidate.tools[1]
    assert isinstance(specialist_tool, AgentTool)
    specialist_data = specialist_tool.to_dict()["data"]["agent"]["init_parameters"]
    assert specialist_data["chat_generator"]["init_parameters"]["model"] == "cheaper"
    assert [tool["data"]["name"] for tool in specialist_data["tools"]] == ["search_documents"]
    assert assets().require_valid_agent(candidate).allowed is True


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
    first = recipe_from_dict(data)
    second = recipe_from_dict(data)
    assert recipe_fingerprint(first) == recipe_fingerprint(second)
    with pytest.raises(ValueError, match="Unsupported candidate recipe kind"):
        recipe_from_dict({"kind": "python", "code": "dangerous()"})


def test_composite_and_registered_structural_recipes():
    reference = reference_agent()
    composite = CompositeRecipe(
        recipes=(ModelSubstitutionRecipe("cheaper"), ToolSelectionRecipe(("search_documents",)))
    )
    candidate = composite.materialize(reference, assets())
    assert candidate.chat_generator.model == "cheaper"
    assert [tool.name for tool in candidate.tools] == ["search_documents"]

    registry = StructuralRecipeRegistry()
    registry.register("short-run", lambda agent, _assets, params: agent.clone(max_agent_steps=params["steps"]))
    registered = RegisteredStructuralRecipe("short-run", {"steps": 3}, registry)
    assert registered.materialize(reference, assets()).max_agent_steps == 3
    restored = recipe_from_dict(registered.to_dict(), registry=registry)
    assert restored.materialize(reference, assets()).max_agent_steps == 3


def test_missing_tools_and_unregistered_structures_are_rejected():
    reference = reference_agent()
    with pytest.raises(ValueError, match="not configured"):
        ToolSelectionRecipe(("unknown",)).materialize(reference, assets())
    with pytest.raises(ValueError, match="not registered"):
        RegisteredStructuralRecipe("unknown", {}, StructuralRecipeRegistry()).materialize(reference, assets())
