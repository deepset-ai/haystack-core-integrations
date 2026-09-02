import pytest
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    HarnessPatch,
    ModelAsset,
    ToolAsset,
)
from haystack_integrations.agent_pack.optimization.assets.model_identity import generator_model_id

MOCK_GENERATOR_TYPE = "haystack.components.generators.chat.mock.MockChatGenerator"


class UnnamedGenerator:
    """A chat generator that exposes no model identifier at all."""

    def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - signature only, the reply is fixed
        return {"replies": []}

    def to_dict(self):
        return {"type": "tests.UnnamedGenerator", "init_parameters": {}}


class AzureStyleGenerator:
    def __init__(self):
        self.azure_deployment = "eu-gpt-deployment"


class HuggingFaceStyleGenerator:
    def __init__(self):
        self.api_params = {"model": "local/llama"}


def test_catalog_looks_assets_up_and_fails_closed():
    catalog = ApprovedAssetCatalog(
        models=[ModelAsset(model_id="approved", provider="p", deployment="d")],
        tools=[ToolAsset(name="search_documents")],
    )

    assert catalog.model(model_id="approved").provider == "p"
    assert catalog.tool(tool_name="search_documents").name == "search_documents"
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        catalog.model(model_id="unapproved")
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        catalog.tool(tool_name="unapproved")


def test_catalog_looks_patches_up_and_fails_closed():
    catalog = ApprovedAssetCatalog(
        models=[],
        tools=[],
        patches=[HarnessPatch(name="reasoning-low", patch={"chat_generator.init_parameters.x": 1})],
    )

    assert catalog.patch(name="reasoning-low").patch == {"chat_generator.init_parameters.x": 1}
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        catalog.patch(name="unapproved")


def test_duplicate_assets_are_rejected():
    with pytest.raises(ValueError, match="Model asset IDs must be unique"):
        ApprovedAssetCatalog(
            models=[
                ModelAsset(model_id="same", provider="p", deployment="d"),
                ModelAsset(model_id="same", provider="q", deployment="e"),
            ],
            tools=[],
        )
    with pytest.raises(ValueError, match="Tool asset names must be unique"):
        ApprovedAssetCatalog(models=[], tools=[ToolAsset(name="same"), ToolAsset(name="same", provider="other")])
    with pytest.raises(ValueError, match="Patch names must be unique"):
        ApprovedAssetCatalog(
            models=[],
            tools=[],
            patches=[HarnessPatch(name="same", patch={"a": 1}), HarnessPatch(name="same", patch={"b": 2})],
        )


@pytest.mark.parametrize(
    "generator,expected",
    [
        (MockChatGenerator(model="plain"), "plain"),
        (AzureStyleGenerator(), "eu-gpt-deployment"),
        (HuggingFaceStyleGenerator(), "local/llama"),
        (UnnamedGenerator(), None),
    ],
)
def test_model_identity_is_resolved_across_generator_conventions(generator, expected):
    """Cost is priced per model, so an Azure deployment or a Hugging Face API generator must still be identifiable."""
    assert generator_model_id(generator=generator) == expected


def test_model_asset_substitutes_the_model_on_the_reference_generator_class():
    asset = ModelAsset(model_id="cheaper", provider="provider", deployment="local")
    built = asset.build_generator(
        reference_generator=MockChatGenerator([ChatMessage.from_assistant("hi")], model="reference")
    )
    assert built.model == "cheaper"
    assert isinstance(built, MockChatGenerator)


def test_model_asset_can_declare_its_own_generator_for_cross_provider_substitution():
    asset = ModelAsset(
        model_id="declared",
        provider="other-provider",
        deployment="eu",
        generator={"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "declared"}},
    )
    built = asset.build_generator(reference_generator=UnnamedGenerator())
    assert isinstance(built, MockChatGenerator)
    assert built.model == "declared"


def test_declared_generator_must_agree_with_the_catalog_identifier():
    asset = ModelAsset(
        model_id="declared",
        provider="other-provider",
        deployment="eu",
        generator={"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "something-else"}},
    )
    with pytest.raises(ValueError, match="must match the generator"):
        asset.build_generator(reference_generator=MockChatGenerator(model="reference"))


def test_substitution_requires_a_declared_generator_when_the_reference_hides_its_model():
    with pytest.raises(ValueError, match="needs an explicit 'generator' configuration"):
        ModelAsset(model_id="cheaper", provider="provider", deployment="local").build_generator(
            reference_generator=UnnamedGenerator()
        )
