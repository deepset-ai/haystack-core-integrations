from typing import Any, Dict, List, Optional, Tuple

import pytest
from haystack.utils import Secret
from haystack_integrations.utils.nvidia import Model, NimBackend
from requests_mock import Mocker


class MockBackend(NimBackend):
    def __init__(self, model: str, api_key: Optional[Secret] = None, model_kwargs: Optional[Dict[str, Any]] = None):
        api_key = api_key or Secret.from_env_var("NVIDIA_API_KEY")
        super().__init__(model, api_url="", api_key=api_key, model_kwargs=model_kwargs or {})

    def embed(self, texts):
        inputs = texts
        data = [[0.1, 0.2, 0.3] for i in range(len(inputs))]
        return data, {"usage": {"total_tokens": 4, "prompt_tokens": 4}}

    def models(self):
        return [Model(id="aa")]

    def generate(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        return (
            ["This is a mocked response."],
            [{"role": "assistant", "usage": {"prompt_tokens": 5, "total_tokens": 10, "completion_tokens": 5}}],
        )


@pytest.fixture
def mock_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "http://localhost:8080/v1/models",
        json={
            "data": [
                {
                    "id": "model1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model1",
                },
            ]
        },
    )
