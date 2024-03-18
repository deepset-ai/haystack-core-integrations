from typing import Any, Dict, List, Optional, Tuple

import requests

from .backend import GeneratorBackend

REQUEST_TIMEOUT = 60


class NimBackend(GeneratorBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(headers)

        self.model = model
        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        # We're using the chat completion endpoint as the local containers don't support
        # the /completions endpoint. So both the non-chat and chat generator will use this.
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
                "finish_reason": choice["finish_reason"],
                "usage": {
                    "prompt_tokens": completions["usage"]["prompt_tokens"],
                    "completion_tokens": completions["usage"]["completion_tokens"],
                    "total_tokens": completions["usage"]["total_tokens"],
                },
            }
            meta.append(choice_meta)

        return replies, meta
