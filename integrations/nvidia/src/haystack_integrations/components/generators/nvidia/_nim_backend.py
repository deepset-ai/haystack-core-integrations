from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack.utils import Secret

from .backend import GeneratorBackend

REQUEST_TIMEOUT = 60


class NimBackend(GeneratorBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_kwargs: Optional[Dict[str, Any]] = None,
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
            timeout=REQUEST_TIMEOUT,
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
