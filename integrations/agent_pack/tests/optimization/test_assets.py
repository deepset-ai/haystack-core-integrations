import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import AgentTool, tool

from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.optimization import ApprovedAssetCatalog, ModelAsset, ToolAsset
from haystack_integrations.agent_pack.optimization.assets.model_identity import generator_model_id

MOCK_GENERATOR_TYPE = "haystack.components.generators.chat.mock.MockChatGenerator"


def modelled_generator(model, responses=None):
    return MockChatGenerator(responses or [ChatMessage.from_assistant("done")], model=model)


@tool
def nested_search_documents(query: str) -> str:
    """Search documents from a delegated Agent."""
    return query


def test_asset_catalog_validates_nested_agent_tools():
    specialist = Agent(chat_generator=modelled_generator("specialist"), tools=[nested_search_documents])
    coordinator = Agent(
        chat_generator=modelled_generator("reference"),
        tools=[AgentTool(agent=specialist, name="retrieval_specialist", description="Retrieve evidence")],
    )
    catalog = ApprovedAssetCatalog(
        models=[
            ModelAsset(model_id="reference", provider="provider", deployment="remote"),
            ModelAsset(model_id="specialist", provider="provider", deployment="local"),
        ],
        tools=[ToolAsset(name="retrieval_specialist"), ToolAsset(name="nested_search_documents")],
    )

    validation = catalog.require_valid_agent(agent=coordinator)
    assert validation.model_ids == ("reference", "specialist")
    assert validation.tool_names == ("nested_search_documents", "retrieval_specialist")

    invalid = ApprovedAssetCatalog(
        models=[ModelAsset(model_id="reference", provider="provider", deployment="remote")],
        tools=[ToolAsset(name="retrieval_specialist")],
    ).validate_agent(agent=coordinator)
    assert invalid.allowed is False
    assert "model_not_approved:specialist" in invalid.violations
    assert "tool_not_approved:nested_search_documents" in invalid.violations


def test_asset_catalog_validates_models_configured_on_hooks():
    agent = Agent(
        chat_generator=modelled_generator("reference"),
        hooks={"after_run": [BackupAnswerHook(chat_generator=modelled_generator("backup"))]},
    )

    validation = ApprovedAssetCatalog(
        models=[ModelAsset(model_id="reference", provider="provider", deployment="remote")], tools=[]
    ).validate_agent(agent=agent)

    assert validation.allowed is False
    assert validation.model_ids == ("backup", "reference")
    assert "model_not_approved:backup" in validation.violations


class UnnamedGenerator:
    """A chat generator that exposes no model identifier at all."""

    def run(self, messages, tools=None, **kwargs):  # noqa: ARG002 - signature only, the reply is fixed
        return {"replies": []}

    def to_dict(self):
        return {"type": "tests.UnnamedGenerator", "init_parameters": {}}


def test_unidentifiable_assets_fail_closed_but_can_be_downgraded_to_warnings():
    agent = Agent(chat_generator=UnnamedGenerator())
    catalog_arguments = {
        "models": [ModelAsset(model_id="reference", provider="provider", deployment="remote")],
        "tools": [],
    }

    strict = ApprovedAssetCatalog(**catalog_arguments).validate_agent(agent=agent)
    assert strict.allowed is False
    assert strict.violations == ("model_not_identifiable",)

    lenient = ApprovedAssetCatalog(**catalog_arguments, strict_identification=False).validate_agent(agent=agent)
    assert lenient.allowed is True
    assert lenient.warnings == ("model_not_identifiable",)


class AzureStyleGenerator:
    def __init__(self):
        self.azure_deployment = "eu-gpt-deployment"


class HuggingFaceStyleGenerator:
    def __init__(self):
        self.api_params = {"model": "local/llama"}


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
    """Azure deployments and Hugging Face API generators must not be rejected as unidentifiable."""
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
