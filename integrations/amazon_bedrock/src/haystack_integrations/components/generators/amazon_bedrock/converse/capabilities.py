from enum import Enum, auto


class ModelCapability(Enum):
    CONVERSE = auto()
    CONVERSE_STREAM = auto()
    SYSTEM_PROMPTS = auto()
    DOCUMENT_CHAT = auto()
    VISION = auto()
    TOOL_USE = auto()
    STREAMING_TOOL_USE = auto()
    GUARDRAILS = auto()


# https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html

MODEL_CAPABILITIES = {
    "ai21.jamba-instruct-.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
    },
    "ai21.j2-.*": {
        ModelCapability.CONVERSE,
        ModelCapability.GUARDRAILS,
    },
    "amazon.titan-text-.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "anthropic.claude-v2.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "anthropic.claude-3.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.VISION,
        ModelCapability.TOOL_USE,
        ModelCapability.STREAMING_TOOL_USE,
        ModelCapability.GUARDRAILS,
    },
    "cohere.command-text.*": {
        ModelCapability.CONVERSE,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "cohere.command-light.*": {
        ModelCapability.CONVERSE,
        ModelCapability.GUARDRAILS,
    },
    "cohere.command-r.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.TOOL_USE,
    },
    "meta.llama2.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "meta.llama3.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "meta.llama3-1.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.TOOL_USE,
        ModelCapability.GUARDRAILS,
    },
    "mistral.mistral-.*-instruct": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.GUARDRAILS,
    },
    "mistral.mistral-large.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.DOCUMENT_CHAT,
        ModelCapability.TOOL_USE,
        ModelCapability.GUARDRAILS,
    },
    "mistral.mistral-small.*": {
        ModelCapability.CONVERSE,
        ModelCapability.CONVERSE_STREAM,
        ModelCapability.SYSTEM_PROMPTS,
        ModelCapability.TOOL_USE,
        ModelCapability.GUARDRAILS,
    },
}
