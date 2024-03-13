from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from haystack.utils.auth import Secret
from haystack_integrations.utils.nvidia import NvidiaCloudFunctionsClient

from .backend import EmbedderBackend

MAX_INPUT_STRING_LENGTH = 2048
MAX_INPUTS = 50


class NvcfBackend(EmbedderBackend):
    def __init__(
        self,
        model: str,
        api_key: Secret,
        batch_size: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not model.startswith("playground_"):
            model = f"playground_{model}"

        super().__init__(model=model, model_kwargs=model_kwargs)

        if batch_size > MAX_INPUTS:
            msg = f"NVIDIA Cloud Functions currently support a maximum batch size of {MAX_INPUTS}."
            raise ValueError(msg)

        self.api_key = api_key
        self.client = NvidiaCloudFunctionsClient(
            api_key=api_key,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self.nvcf_id = self.client.get_model_nvcf_id(self.model_name)

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        request = EmbeddingsRequest(input=texts, **self.model_kwargs).to_dict()
        json_response = self.client.query_function(self.nvcf_id, request)
        response = EmbeddingsResponse.from_dict(json_response)

        # Sort resulting embeddings by index
        assert all(isinstance(r.embedding, list) for r in response.data)
        sorted_embeddings: List[List[float]] = [r.embedding for r in sorted(response.data, key=lambda e: e.index)]  # type: ignore
        metadata = {"usage": response.usage.to_dict()}
        return sorted_embeddings, metadata


@dataclass
class EmbeddingsRequest:
    input: Union[str, List[str]]
    model: Literal["query", "passage"]
    encoding_format: Literal["float", "base64"] = "float"

    def __post_init__(self):
        if isinstance(self.input, list):
            if len(self.input) > MAX_INPUTS:
                msg = f"The number of inputs should not exceed {MAX_INPUTS}"
                raise ValueError(msg)
        else:
            self.input = [self.input]

        if len(self.input) == 0:
            msg = "The number of inputs should not be 0"
            raise ValueError(msg)

        if any(len(x) > MAX_INPUT_STRING_LENGTH for x in self.input):
            msg = f"The length of each input should not exceed {MAX_INPUT_STRING_LENGTH} characters"
            raise ValueError(msg)

        if self.encoding_format not in ["float", "base64"]:
            msg = "encoding_format should be either 'float' or 'base64'"
            raise ValueError(msg)

        if self.model not in ["query", "passage"]:
            msg = "model should be either 'query' or 'passage'"
            raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Usage:
    prompt_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Embeddings:
    index: int
    embedding: Union[List[float], str]


@dataclass
class EmbeddingsResponse:
    data: List[Embeddings]
    usage: Usage

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingsResponse":
        try:
            embeddings = [Embeddings(**x) for x in data["data"]]
            usage = Usage(**data["usage"])
            return cls(data=embeddings, usage=usage)
        except (KeyError, TypeError) as e:
            msg = f"Failed to parse EmbeddingsResponse from data: {data}"
            raise ValueError(msg) from e
