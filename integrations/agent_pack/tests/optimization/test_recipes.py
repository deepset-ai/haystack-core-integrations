import pytest
from haystack import Document
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool
from pydantic import ValidationError

from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.optimization import (
    ApplyPatchRecipe,
    ApprovedAssetCatalog,
    HarnessPatch,
    ModelAsset,
    ModelSubstitutionRecipe,
    SystemPromptRecipe,
    ToolAsset,
)
from haystack_integrations.agent_pack.optimization.recipes import parse_proposal, recipe_fingerprint

TOP_K_PATH = "tools.search_documents.component.init_parameters.top_k"
REASONING_PATH = "chat_generator.init_parameters.generation_kwargs.reasoning.effort"


def retrieval_tool(top_k=1):
    store = InMemoryDocumentStore()
    store.write_documents([Document(content="CRISPR gene editing")])
    return ComponentTool(
        component=InMemoryBM25Retriever(document_store=store, top_k=top_k),
        name="search_documents",
        description="Search documents.",
    )


def assets(patches=None):
    return ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="provider", deployment="remote"),
            ModelAsset(model_id="cheaper", provider="provider", deployment="local"),
        ],
        tools=[ToolAsset(name="search_documents")],
        patches=patches
        or [
            HarnessPatch(name="retrieval-top-3", patch={TOP_K_PATH: 3}, description="Retrieve more documents."),
            HarnessPatch(name="reasoning-high", patch={REASONING_PATH: "high"}, description="Think harder."),
        ],
    )


def reference_agent(**kwargs):
    defaults = {
        "chat_generator": MockChatGenerator(model="reference"),
        "tools": [retrieval_tool()],
        "system_prompt": "reference prompt",
    }
    return Agent(**{**defaults, **kwargs})


def test_model_and_prompt_recipes_clone_without_mutating_the_reference():
    reference = reference_agent()

    substituted = ModelSubstitutionRecipe(model_id="cheaper").materialize(reference=reference, assets=assets())
    prompted = SystemPromptRecipe(system_prompt="candidate prompt").materialize(reference=reference, assets=assets())

    assert substituted.chat_generator.model == "cheaper"
    assert substituted.chat_generator is not reference.chat_generator
    assert prompted.system_prompt == "candidate prompt"
    assert reference.chat_generator.model == "reference"
    assert reference.system_prompt == "reference prompt"


def test_candidates_do_not_share_mutable_containers_with_the_reference():
    """A candidate that registers a hook or appends a tool must not change the reference harness."""
    reference = reference_agent(hooks={"after_run": [BackupAnswerHook(chat_generator=MockChatGenerator("backup"))]})

    candidate = ModelSubstitutionRecipe(model_id="cheaper").materialize(reference=reference, assets=assets())
    candidate.hooks["after_run"].append(BackupAnswerHook(chat_generator=MockChatGenerator("extra")))
    candidate.hooks["before_tool"] = []
    candidate.tools.append(retrieval_tool())
    candidate.state_schema["injected"] = {"type": str}

    assert candidate.hooks is not reference.hooks
    assert len(reference.hooks["after_run"]) == 1
    assert "before_tool" not in reference.hooks
    assert len(reference.tools) == 1
    assert "injected" not in reference.state_schema


def test_a_patch_reaches_a_tool_parameter_and_keeps_the_documents():
    """A patch goes through the serialized harness, which is how it reaches a component behind a tool."""
    reference = reference_agent()

    candidate = ApplyPatchRecipe(patch="retrieval-top-3").materialize(reference=reference, assets=assets())

    assert candidate.tools[0]._component.top_k == 3
    assert reference.tools[0]._component.top_k == 1
    # The rebuilt store reconnects to the same shared storage, so the candidate retrieves from the same corpus.
    assert candidate.tools[0]._component.document_store.count_documents() == 1


def test_a_patch_creates_missing_intermediate_settings(monkeypatch):
    """Reasoning effort has to be settable on a generator that declares no generation parameters at all."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    reference = reference_agent(chat_generator=OpenAIChatGenerator(model="gpt-5"))

    candidate = ApplyPatchRecipe(patch="reasoning-high").materialize(reference=reference, assets=assets())

    assert candidate.chat_generator.generation_kwargs == {"reasoning": {"effort": "high"}}
    assert reference.chat_generator.generation_kwargs == {}


def test_only_approved_patches_can_be_applied():
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        ApplyPatchRecipe(patch="unapproved").materialize(reference=reference_agent(), assets=assets())


def test_a_patch_path_that_does_not_resolve_is_reported():
    patches = [HarnessPatch(name="missing-tool", patch={"tools.absent.component.init_parameters.top_k": 3})]
    with pytest.raises(ValueError, match="names 'absent', which is not in the harness"):
        ApplyPatchRecipe(patch="missing-tool").materialize(reference=reference_agent(), assets=assets(patches=patches))


def test_a_harness_that_does_not_serialize_cannot_be_patched():
    class Unserializable:
        def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - signature only
            return {"replies": []}

        def to_dict(self):
            msg = "nope"
            raise TypeError(msg)

    reference = Agent(chat_generator=Unserializable())
    with pytest.raises(ValueError, match="does not serialize, so it cannot be patched"):
        ApplyPatchRecipe(patch="reasoning-high").materialize(reference=reference, assets=assets())


def test_an_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="non-empty prompt"):
        SystemPromptRecipe(system_prompt="   ")


def test_fingerprints_are_stable_for_equivalent_recipes():
    first = SystemPromptRecipe(system_prompt="candidate prompt")
    second = SystemPromptRecipe(system_prompt="candidate prompt")
    assert recipe_fingerprint(recipe=first) == recipe_fingerprint(recipe=second)
    assert recipe_fingerprint(recipe=ApplyPatchRecipe(patch="reasoning-high")) != recipe_fingerprint(
        recipe=ApplyPatchRecipe(patch="retrieval-top-3")
    )


def test_proposals_are_validated_against_the_catalog():
    """The catalog is the compliance boundary: an unapproved choice cannot survive validation."""
    recipes = parse_proposal(
        payload={
            "recipes": [
                {"kind": "model_substitution", "model_id": "cheaper"},
                {"kind": "apply_patch", "patch": "reasoning-high"},
                {"kind": "system_prompt", "system_prompt": "try this"},
            ]
        },
        assets=assets(),
    )
    assert recipes == [
        ModelSubstitutionRecipe(model_id="cheaper"),
        ApplyPatchRecipe(patch="reasoning-high"),
        SystemPromptRecipe(system_prompt="try this"),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"recipes": [{"kind": "python", "code": "dangerous()"}]}, id="unknown-kind"),
        pytest.param({"recipes": [{"kind": "model_substitution", "model_id": "unapproved"}]}, id="unapproved-model"),
        pytest.param({"recipes": [{"kind": "apply_patch", "patch": "unapproved"}]}, id="unapproved-patch"),
        pytest.param({"recipes": [{"kind": "system_prompt", "system_prompt": ""}]}, id="empty-prompt"),
        pytest.param(
            {"recipes": [{"kind": "model_substitution", "model_id": "cheaper", "extra": 1}]}, id="unexpected-field"
        ),
        pytest.param({"recipes": [{"kind": "tool_selection", "tool_names": ["x"]}]}, id="removed-kind"),
        pytest.param({"recipes": [{"kind": "apply_patch", "patch": {"some.path": 1}}]}, id="optimizer-authored-patch"),
    ],
)
def test_invalid_proposals_are_rejected(payload):
    with pytest.raises(ValidationError):
        parse_proposal(payload=payload, assets=assets())


def test_too_many_proposals_are_rejected():
    payload = {"recipes": [{"kind": "model_substitution", "model_id": "cheaper"}] * 3}
    with pytest.raises(ValidationError):
        parse_proposal(payload=payload, assets=assets(), max_recipes=2)


def test_an_empty_proposal_is_valid():
    """Having nothing worth trying is a legitimate answer, not a malformed one."""
    assert parse_proposal(payload={"recipes": []}, assets=assets()) == []
