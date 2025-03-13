from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Model:
    """
    Model information.

    id: unique identifier for the model, passed as model parameter for requests
    model_type: API type (chat, vlm, embedding, ranking, completions)
    client: client name, e.g. NvidiaGenerator, NVIDIAEmbeddings,
            NVIDIARerank, NvidiaTextEmbedder, NvidiaDocumentEmbedder
    endpoint: custom endpoint for the model
    aliases: list of aliases for the model

    All aliases are deprecated and will trigger a warning when used.
    """

    id: str
    model_type: Optional[Literal["chat", "embedding", "ranking"]] = None
    client: Optional[Literal["NvidiaGenerator", "NvidiaTextEmbedder", "NvidiaDocumentEmbedder", "NvidiaRanker"]] = None
    endpoint: Optional[str] = None
    aliases: Optional[list] = None
    base_model: Optional[str] = None
    supports_tools: Optional[bool] = False
    supports_structured_output: Optional[bool] = False

    def __hash__(self) -> int:
        return hash(self.id)

    def validate(self):
        if self.client:
            supported = {
                "NvidiaGenerator": ("chat",),
                "NvidiaTextEmbedder": ("embedding",),
                "NvidiaDocumentEmbedder": ("embedding",),
                "NvidiaRanker": ("ranking",),
            }
            model_type = self.model_type
            if model_type not in supported[self.client]:
                err_msg = f"Model type '{model_type}' not supported by client '{self.client}'"
                raise ValueError(err_msg)

        return hash(self.id)


DEFAULT_API_URL = "https://integrate.api.nvidia.com/v1"

CHAT_MODEL_TABLE = {
    "meta/codellama-70b": Model(
        id="meta/codellama-70b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=[
            "ai-codellama-70b",
            "playground_llama2_code_70b",
            "llama2_code_70b",
            "playground_llama2_code_34b",
            "llama2_code_34b",
            "playground_llama2_code_13b",
            "llama2_code_13b",
        ],
    ),
    "google/gemma-7b": Model(
        id="google/gemma-7b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-gemma-7b", "playground_gemma_7b", "gemma_7b"],
    ),
    "meta/llama2-70b": Model(
        id="meta/llama2-70b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=[
            "ai-llama2-70b",
            "playground_llama2_70b",
            "llama2_70b",
            "playground_llama2_13b",
            "llama2_13b",
        ],
    ),
    "mistralai/mistral-7b-instruct-v0.2": Model(
        id="mistralai/mistral-7b-instruct-v0.2",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-mistral-7b-instruct-v2", "playground_mistral_7b", "mistral_7b"],
    ),
    "mistralai/mixtral-8x7b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x7b-instruct-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-mixtral-8x7b-instruct", "playground_mixtral_8x7b", "mixtral_8x7b"],
    ),
    "google/codegemma-7b": Model(
        id="google/codegemma-7b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-codegemma-7b"],
    ),
    "google/gemma-2b": Model(
        id="google/gemma-2b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-gemma-2b", "playground_gemma_2b", "gemma_2b"],
    ),
    "google/recurrentgemma-2b": Model(
        id="google/recurrentgemma-2b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-recurrentgemma-2b"],
    ),
    "mistralai/mistral-large": Model(
        id="mistralai/mistral-large",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-mistral-large"],
    ),
    "mistralai/mixtral-8x22b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x22b-instruct-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-mixtral-8x22b-instruct"],
    ),
    "meta/llama3-8b-instruct": Model(
        id="meta/llama3-8b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-llama3-8b"],
    ),
    "meta/llama3-70b-instruct": Model(
        id="meta/llama3-70b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-llama3-70b"],
    ),
    "microsoft/phi-3-mini-128k-instruct": Model(
        id="microsoft/phi-3-mini-128k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-mini"],
    ),
    "snowflake/arctic": Model(
        id="snowflake/arctic",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-arctic"],
    ),
    "databricks/dbrx-instruct": Model(
        id="databricks/dbrx-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-dbrx-instruct"],
    ),
    "microsoft/phi-3-mini-4k-instruct": Model(
        id="microsoft/phi-3-mini-4k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-mini-4k", "playground_phi2", "phi2"],
    ),
    "seallms/seallm-7b-v2.5": Model(
        id="seallms/seallm-7b-v2.5",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-seallm-7b"],
    ),
    "aisingapore/sea-lion-7b-instruct": Model(
        id="aisingapore/sea-lion-7b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-sea-lion-7b-instruct"],
    ),
    "microsoft/phi-3-small-8k-instruct": Model(
        id="microsoft/phi-3-small-8k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-small-8k-instruct"],
    ),
    "microsoft/phi-3-small-128k-instruct": Model(
        id="microsoft/phi-3-small-128k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-small-128k-instruct"],
    ),
    "microsoft/phi-3-medium-4k-instruct": Model(
        id="microsoft/phi-3-medium-4k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-medium-4k-instruct"],
    ),
    "ibm/granite-8b-code-instruct": Model(
        id="ibm/granite-8b-code-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-granite-8b-code-instruct"],
    ),
    "ibm/granite-34b-code-instruct": Model(
        id="ibm/granite-34b-code-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-granite-34b-code-instruct"],
    ),
    "google/codegemma-1.1-7b": Model(
        id="google/codegemma-1.1-7b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-codegemma-1.1-7b"],
    ),
    "mediatek/breeze-7b-instruct": Model(
        id="mediatek/breeze-7b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-breeze-7b-instruct"],
    ),
    "upstage/solar-10.7b-instruct": Model(
        id="upstage/solar-10.7b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-solar-10_7b-instruct"],
    ),
    "writer/palmyra-med-70b-32k": Model(
        id="writer/palmyra-med-70b-32k",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-palmyra-med-70b-32k"],
    ),
    "writer/palmyra-med-70b": Model(
        id="writer/palmyra-med-70b",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-palmyra-med-70b"],
    ),
    "mistralai/mistral-7b-instruct-v0.3": Model(
        id="mistralai/mistral-7b-instruct-v0.3",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-mistral-7b-instruct-v03"],
    ),
    "01-ai/yi-large": Model(
        id="01-ai/yi-large",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-yi-large"],
    ),
    "nvidia/nemotron-4-340b-instruct": Model(
        id="nvidia/nemotron-4-340b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["qa-nemotron-4-340b-instruct"],
    ),
    "mistralai/codestral-22b-instruct-v0.1": Model(
        id="mistralai/codestral-22b-instruct-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-codestral-22b-instruct-v01"],
        supports_structured_output=True,
    ),
    "google/gemma-2-9b-it": Model(
        id="google/gemma-2-9b-it",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-gemma-2-9b-it"],
    ),
    "google/gemma-2-27b-it": Model(
        id="google/gemma-2-27b-it",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-gemma-2-27b-it"],
    ),
    "microsoft/phi-3-medium-128k-instruct": Model(
        id="microsoft/phi-3-medium-128k-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-phi-3-medium-128k-instruct"],
    ),
    "deepseek-ai/deepseek-coder-6.7b-instruct": Model(
        id="deepseek-ai/deepseek-coder-6.7b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        aliases=["ai-deepseek-coder-6_7b-instruct"],
    ),
    "nv-mistralai/mistral-nemo-12b-instruct": Model(
        id="nv-mistralai/mistral-nemo-12b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-8b-instruct": Model(
        id="meta/llama-3.1-8b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-70b-instruct": Model(
        id="meta/llama-3.1-70b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "meta/llama-3.1-405b-instruct": Model(
        id="meta/llama-3.1-405b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "nvidia/usdcode-llama3-70b-instruct": Model(
        id="nvidia/usdcode-llama3-70b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "mistralai/mamba-codestral-7b-v0.1": Model(
        id="mistralai/mamba-codestral-7b-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "writer/palmyra-fin-70b-32k": Model(
        id="writer/palmyra-fin-70b-32k",
        model_type="chat",
        client="NvidiaGenerator",
        supports_structured_output=True,
    ),
    "google/gemma-2-2b-it": Model(
        id="google/gemma-2-2b-it",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "mistralai/mistral-large-2-instruct": Model(
        id="mistralai/mistral-large-2-instruct",
        model_type="chat",
        client="NvidiaGenerator",
        supports_tools=True,
        supports_structured_output=True,
    ),
    "mistralai/mathstral-7b-v0.1": Model(
        id="mistralai/mathstral-7b-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "rakuten/rakutenai-7b-instruct": Model(
        id="rakuten/rakutenai-7b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "rakuten/rakutenai-7b-chat": Model(
        id="rakuten/rakutenai-7b-chat",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "baichuan-inc/baichuan2-13b-chat": Model(
        id="baichuan-inc/baichuan2-13b-chat",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "thudm/chatglm3-6b": Model(
        id="thudm/chatglm3-6b",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "microsoft/phi-3.5-mini-instruct": Model(
        id="microsoft/phi-3.5-mini-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "microsoft/phi-3.5-moe-instruct": Model(
        id="microsoft/phi-3.5-moe-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "nvidia/nemotron-mini-4b-instruct": Model(
        id="nvidia/nemotron-mini-4b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "ai21labs/jamba-1.5-large-instruct": Model(
        id="ai21labs/jamba-1.5-large-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "ai21labs/jamba-1.5-mini-instruct": Model(
        id="ai21labs/jamba-1.5-mini-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "yentinglin/llama-3-taiwan-70b-instruct": Model(
        id="yentinglin/llama-3-taiwan-70b-instruct",
        model_type="chat",
        client="NvidiaGenerator",
    ),
    "tokyotech-llm/llama-3-swallow-70b-instruct-v0.1": Model(
        id="tokyotech-llm/llama-3-swallow-70b-instruct-v0.1",
        model_type="chat",
        client="NvidiaGenerator",
    ),
}

EMBEDDING_MODEL_TABLE = {
    "snowflake/arctic-embed-l": Model(
        id="snowflake/arctic-embed-l",
        model_type="embedding",
        aliases=["ai-arctic-embed-l"],
    ),
    "NV-Embed-QA": Model(
        id="NV-Embed-QA",
        model_type="embedding",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia",
        aliases=[
            "ai-embed-qa-4",
            "playground_nvolveqa_40k",
            "nvolveqa_40k",
        ],
    ),
    "nvidia/nv-embed-v1": Model(
        id="nvidia/nv-embed-v1",
        model_type="embedding",
        aliases=["ai-nv-embed-v1"],
    ),
    "nvidia/nv-embedqa-mistral-7b-v2": Model(
        id="nvidia/nv-embedqa-mistral-7b-v2",
        model_type="embedding",
    ),
    "nvidia/nv-embedqa-e5-v5": Model(
        id="nvidia/nv-embedqa-e5-v5",
        model_type="embedding",
    ),
    "baai/bge-m3": Model(
        id="baai/bge-m3",
        model_type="embedding",
    ),
    "nvidia/embed-qa-4": Model(
        id="nvidia/embed-qa-4",
        model_type="embedding",
    ),
    "nvidia/llama-3.2-nv-embedqa-1b-v1": Model(
        id="nvidia/llama-3.2-nv-embedqa-1b-v1",
        model_type="embedding",
    ),
    "nvidia/llama-3.2-nv-embedqa-1b-v2": Model(
        id="nvidia/llama-3.2-nv-embedqa-1b-v2",
        model_type="embedding",
    ),
}
RANKING_MODEL_TABLE = {
    "nv-rerank-qa-mistral-4b:1": Model(
        id="nv-rerank-qa-mistral-4b:1",
        model_type="ranking",
        client="NvidiaRanker",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        aliases=["ai-rerank-qa-mistral-4b"],
    ),
    "nvidia/nv-rerankqa-mistral-4b-v3": Model(
        id="nvidia/nv-rerankqa-mistral-4b-v3",
        model_type="ranking",
        client="NvidiaRanker",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking",
    ),
    "nvidia/llama-3.2-nv-rerankqa-1b-v1": Model(
        id="nvidia/llama-3.2-nv-rerankqa-1b-v1",
        model_type="ranking",
        client="NvidiaRanker",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v1/reranking",
    ),
    "nvidia/llama-3.2-nv-rerankqa-1b-v2": Model(
        id="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        model_type="ranking",
        client="NvidiaRanker",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
    ),
}

DEFAULT_MODELS = {
    "embedding": "nvidia/nv-embedqa-e5-v5",
    "ranking": "nvidia/nv-rerankqa-mistral-4b-v3",
    "chat": "meta/llama3-8b-instruct",
}


MODEL_TABLE = {**CHAT_MODEL_TABLE, **EMBEDDING_MODEL_TABLE, **RANKING_MODEL_TABLE}
