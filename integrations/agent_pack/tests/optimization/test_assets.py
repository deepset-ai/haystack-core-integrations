import pytest
from haystack.components.generators.chat import MockChatGenerator

from haystack_integrations.agent_pack.optimization import (
    ApprovedAssetCatalog,
    HarnessPatch,
    ModelAsset,
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


def test_catalog_looks_models_up_and_fails_closed():
    catalog = ApprovedAssetCatalog(models=[ModelAsset(model_id="approved", input_cost_per_million=2.0)])

    assert catalog.model(model_id="approved").input_cost_per_million == 2.0
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        catalog.model(model_id="unapproved")


def test_catalog_looks_patches_up_and_fails_closed():
    catalog = ApprovedAssetCatalog(
        models=[],
        patches=[HarnessPatch(name="reasoning-low", patch={"chat_generator.init_parameters.x": 1})],
    )

    assert catalog.patch(name="reasoning-low").patch == {"chat_generator.init_parameters.x": 1}
    with pytest.raises(ValueError, match="not in the approved asset catalog"):
        catalog.patch(name="unapproved")


def test_duplicate_assets_are_rejected():
    with pytest.raises(ValueError, match="Model asset IDs must be unique"):
        ApprovedAssetCatalog(
            models=[
                ModelAsset(model_id="same"),
                ModelAsset(model_id="same"),
            ],
        )
    with pytest.raises(ValueError, match="Patch names must be unique"):
        ApprovedAssetCatalog(
            models=[],
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


def test_model_asset_patches_the_model_identifier_where_the_generator_keeps_it():
    """The key differs by provider, so it is read from the reference generator rather than assumed."""
    asset = ModelAsset(model_id="cheaper")

    plain = asset.substitution_patch(serialized_generator={"type": "X", "init_parameters": {"model": "reference"}})
    azure = asset.substitution_patch(
        serialized_generator={"type": "X", "init_parameters": {"azure_deployment": "reference"}}
    )
    hugging_face = asset.substitution_patch(
        serialized_generator={"type": "X", "init_parameters": {"api_params": {"model": "reference"}}}
    )

    assert plain == {"chat_generator.init_parameters.model": "cheaper"}
    assert azure == {"chat_generator.init_parameters.azure_deployment": "cheaper"}
    assert hugging_face == {"chat_generator.init_parameters.api_params.model": "cheaper"}


def test_model_asset_can_declare_a_whole_generator_for_cross_provider_substitution():
    declared = {"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "declared"}}
    asset = ModelAsset(model_id="declared", generator=declared)

    patch = asset.substitution_patch(serialized_generator={"type": "X", "init_parameters": {}})

    assert patch == {"chat_generator": declared}


def test_declared_generator_must_agree_with_the_catalog_identifier():
    asset = ModelAsset(
        model_id="declared",
        generator={"type": MOCK_GENERATOR_TYPE, "init_parameters": {"model": "something-else"}},
    )
    with pytest.raises(ValueError, match="must match the generator"):
        asset.substitution_patch(serialized_generator={"type": "X", "init_parameters": {"model": "reference"}})


def test_substitution_requires_a_declared_generator_when_the_reference_hides_its_model():
    asset = ModelAsset(model_id="cheaper")
    with pytest.raises(ValueError, match="needs an explicit 'generator' configuration"):
        asset.substitution_patch(serialized_generator={"type": "X", "init_parameters": {}})
