from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Union

from haystack_integrations.utils.nvidia import NvidiaCloudFunctionsClient

from .models import NvidiaEmbeddingModel

MAX_INPUT_STRING_LENGTH = 2048
MAX_INPUTS = 50


def get_model_nvcf_id(model: NvidiaEmbeddingModel, client: NvidiaCloudFunctionsClient) -> str:
    """
    Returns the Nvidia Cloud Functions UUID for the given model.
    """

    available_functions = client.available_functions()
    func = available_functions.get(str(model))
    if func is None:
        msg = f"Model '{model}' was not found on the Nvidia Cloud Functions backend"
        raise ValueError(msg)
    elif func.status != "ACTIVE":
        msg = f"Model '{model}' is not currently active/usable on the Nvidia Cloud Functions backend"
        raise ValueError(msg)

    return func.id


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
