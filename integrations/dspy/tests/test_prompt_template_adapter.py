import sys

sys.path.insert(0, "src/haystack_integrations/components/generators/dspy")
from prompt_template_adapter import PromptTemplateAdapter


def test_mistral_wraps_correctly():
    adapter = PromptTemplateAdapter(model_family="mistral")
    assert adapter.wrap("Hello") == "[INST] Hello [/INST]"


def test_llama_wraps_correctly():
    adapter = PromptTemplateAdapter(model_family="llama")
    assert adapter.wrap("Hello") == "<s>[INST] Hello [/INST]"


def test_default_is_passthrough():
    adapter = PromptTemplateAdapter()
    assert adapter.wrap("Hello") == "Hello"


def test_unknown_model_falls_back_to_default():
    adapter = PromptTemplateAdapter(model_family="unknown_model")
    assert adapter.wrap("Hello") == "Hello"
