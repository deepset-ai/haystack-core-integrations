from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.utils.nvidia.utils import REQUEST_TIMEOUT, Model


class NimBackend:
    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        if api_key:
            headers["authorization"] = f"Bearer {api_key.resolve_value()}"

        self.session = requests.Session()
        self.session.headers.update(headers)

        self.model = model
        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

        if timeout is None:
            timeout = REQUEST_TIMEOUT
        self.timeout = timeout

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        url = f"{self.api_url}/embeddings"

        res = self.session.post(
            url,
            json={
                "model": self.model,
                "input": texts,
                **self.model_kwargs,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()

        data = res.json()
        # Sort the embeddings by index, we don't know whether they're out of order or not
        embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda e: e["index"])]

        return embeddings, {"usage": data["usage"]}

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        # We're using the chat completion endpoint as the NIM API doesn't support
        # the /completions endpoint. So both the non-chat and chat generator will use this.
        # This is the same for local containers and the cloud API.
        url = f"{self.api_url}/chat/completions"

        res = self.session.post(
            url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                **self.model_kwargs,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()

        completions = res.json()
        choices = completions["choices"]
        # Sort the choices by index, we don't know whether they're out of order or not
        choices.sort(key=lambda c: c["index"])
        replies = []
        meta = []
        for choice in choices:
            message = choice["message"]
            replies.append(message["content"])
            choice_meta = {
                "role": message["role"],
                "usage": {
                    "prompt_tokens": completions["usage"]["prompt_tokens"],
                    "total_tokens": completions["usage"]["total_tokens"],
                },
            }
            # These fields could be null, the others will always be present
            if "finish_reason" in choice:
                choice_meta["finish_reason"] = choice["finish_reason"]
            if "completion_tokens" in completions["usage"]:
                choice_meta["usage"]["completion_tokens"] = completions["usage"]["completion_tokens"]

            meta.append(choice_meta)

        return replies, meta

    def models(self) -> List[Model]:
        url = f"{self.api_url}/models"

        res = self.session.get(
            url,
            timeout=self.timeout,
        )
        res.raise_for_status()

        data = res.json()["data"]
        models = [Model(element["id"]) for element in data if "id" in element]
        if not models:
            msg = f"No hosted model were found at URL '{url}'."
            raise ValueError(msg)
        return models

    def rank(
        self,
        query: str,
        documents: List[Document],
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        url = endpoint or f"{self.api_url}/ranking"

        res = self.session.post(
            url,
            json={
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": doc.content} for doc in documents],
                **self.model_kwargs,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()

        data = res.json()
        assert "rankings" in data, f"Expected 'rankings' in response, got {data}"

        return data["rankings"]
