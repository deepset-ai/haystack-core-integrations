import os

import pytest
from haystack.utils import Secret

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

MODELS_TO_TEST = [
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]


def _generator(model: str) -> AmazonBedrockGenerator:
    return AmazonBedrockGenerator(
        model=model,
        max_length=64,
        aws_region_name=Secret.from_token(os.environ["AWS_REGION"]),
    )


def _assert_usage(usage: dict) -> None:
    assert isinstance(usage["input_tokens"], int) and usage["input_tokens"] > 0
    assert isinstance(usage["output_tokens"], int) and usage["output_tokens"] > 0
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("AWS_BEARER_TOKEN_BEDROCK") or not os.getenv("AWS_REGION"),
    reason="AWS_BEARER_TOKEN_BEDROCK and AWS_REGION must be set",
)
class TestAmazonBedrockGeneratorInference:
    @pytest.mark.parametrize("model", MODELS_TO_TEST)
    def test_run_non_streaming_normalizes_usage(self, model: str) -> None:
        generator = _generator(model)
        result = generator.run("What is the capital of France? Reply in one word.")

        assert result["replies"], "No replies received"
        assert isinstance(result["replies"][0], str) and result["replies"][0]

        meta = result["meta"]
        assert "usage" in meta, f"meta does not contain a normalized 'usage' block: {meta}"
        _assert_usage(meta["usage"])

    @pytest.mark.parametrize("model", MODELS_TO_TEST)
    def test_run_streaming_normalizes_usage(self, model: str) -> None:
        generator = _generator(model)
        chunks: list = []
        result = generator.run(
            "What is the capital of France? Reply in one word.",
            streaming_callback=chunks.append,
        )

        assert chunks, "Streaming callback was not invoked"
        assert result["replies"], "No replies received"
        assert isinstance(result["replies"][0], str) and result["replies"][0]

        meta = result["meta"]
        assert "usage" in meta, f"meta does not contain a normalized 'usage' block: {meta}"
        _assert_usage(meta["usage"])
