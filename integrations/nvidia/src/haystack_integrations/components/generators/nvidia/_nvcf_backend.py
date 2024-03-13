from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from haystack.utils.auth import Secret
from haystack_integrations.utils.nvidia import NvidiaCloudFunctionsClient

from .backend import GeneratorBackend

MAX_INPUT_STRING_LENGTH = 2048
MAX_INPUTS = 50


class NvcfBackend(GeneratorBackend):
    def __init__(
        self,
        model: str,
        api_key: Secret,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not model.startswith("playground_"):
            model = f"playground_{model}"

        super().__init__(model=model, model_kwargs=model_kwargs)

        self.api_key = api_key
        self.client = NvidiaCloudFunctionsClient(
            api_key=api_key,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self.nvcf_id = self.client.get_model_nvcf_id(self.model_name)

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        messages = [Message(role="user", content=prompt)]
        request = GenerationRequest(messages=messages, **self.model_kwargs).to_dict()
        json_response = self.client.query_function(self.nvcf_id, request)
        response = GenerationResponse.from_dict(json_response)

        replies = []
        meta = []
        for choice in response.choices:
            replies.append(choice.message.content)
            meta.append(
                {
                    "role": choice.message.role,
                    "finish_reason": choice.finish_reason,
                }
            )
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return replies, meta, usage


@dataclass
class Message:
    content: str
    role: str


@dataclass
class GenerationRequest:
    messages: List[Message]
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1024
    seed: Optional[int] = None
    bad: Optional[List[str]] = None
    stop: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str


@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


@dataclass
class GenerationResponse:
    id: str
    choices: List[Choice]
    usage: Usage

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationResponse":
        try:
            return cls(
                id=data["id"],
                choices=[
                    Choice(
                        index=choice["index"],
                        message=Message(content=choice["message"]["content"], role=choice["message"]["role"]),
                        finish_reason=choice["finish_reason"],
                    )
                    for choice in data["choices"]
                ],
                usage=Usage(
                    completion_tokens=data["usage"]["completion_tokens"],
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    total_tokens=data["usage"]["total_tokens"],
                ),
            )
        except (KeyError, TypeError) as e:
            msg = f"Failed to parse {cls.__name__} from data: {data}"
            raise ValueError(msg) from e
