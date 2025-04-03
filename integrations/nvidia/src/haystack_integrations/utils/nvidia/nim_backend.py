# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import requests
from haystack import logging
from haystack.utils import Secret

from .models import DEFAULT_MODELS, Model
from .utils import determine_model, is_hosted, validate_hosted_model

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60.0


class NimBackend:
    def __init__(
        self,
        model: str,
        api_url: str,
        model_type: Optional[Literal["chat", "embedding", "ranking"]] = None,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_kwargs: Optional[Dict[str, Any]] = None,
        client: Optional[
            Literal["NvidiaGenerator", "NvidiaTextEmbedder", "NvidiaDocumentEmbedder", "NvidiaRanker"]
        ] = None,
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

        self.api_url = api_url
        if is_hosted(self.api_url):
            if not api_key:
                warnings.warn(
                    "An API key is required for the hosted NIM. This will become an error in the future.",
                    UserWarning,
                    stacklevel=2,
                )
            if not model and model_type:  # manually set default model
                model = DEFAULT_MODELS[model_type]

            model = validate_hosted_model(model, client)
            if isinstance(model, Model) and model.endpoint:
                # we override the endpoint to use the custom endpoint
                self.api_url = model.endpoint
                self.model_type = model.model_type

        self.model = model.id if isinstance(model, Model) else model
        self.model_kwargs = model_kwargs or {}
        self.client = client
        self.model_type = model_type
        if timeout is None:
            timeout = float(os.environ.get("NVIDIA_TIMEOUT", REQUEST_TIMEOUT))
        self.timeout = timeout

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        url = f"{self.api_url}/embeddings"

        try:
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
        except requests.HTTPError as e:
            logger.error("Error when calling NIM embedding endpoint: Error - {error}", error=e.response.text)
            msg = f"Failed to query embedding endpoint: Error - {e.response.text}"
            raise ValueError(msg) from e

        data = res.json()
        # Sort the embeddings by index, we don't know whether they're out of order or not
        embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda e: e["index"])]

        return embeddings, {"usage": data["usage"]}

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        # We're using the chat completion endpoint as the NIM API doesn't support
        # the /completions endpoint. So both the non-chat and chat generator will use this.
        # This is the same for local containers and the cloud API.
        url = f"{self.api_url}/chat/completions"

        try:
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
        except requests.HTTPError as e:
            logger.error("Error when calling NIM chat completion endpoint: Error - {error}", error=e.response.text)
            msg = f"Failed to query chat completion endpoint: Error - {e.response.text}"
            raise ValueError(msg) from e

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
        models = []
        for element in data:
            assert "id" in element, f"No id found in {element}"
            if not (model := determine_model(element["id"])):
                model = Model(id=element["id"], client=self.client, model_type=self.model_type)
            model.base_model = element.get("root")
            models.append(model)
        if not models:
            logger.error("No hosted model were found at URL '{u}'.", u=url)
            msg = f"No hosted model were found at URL '{url}'."
            raise ValueError(msg)
        return models

    def rank(self, query_text: str, document_texts: List[str]) -> List[Dict[str, Any]]:
        url = self.api_url

        try:
            res = self.session.post(
                url,
                json={
                    "model": self.model,
                    "query": {"text": query_text},
                    "passages": [{"text": text} for text in document_texts],
                    **self.model_kwargs,
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            logger.error("Error when calling NIM ranking endpoint: Error - {error}", error=e.response.text)
            msg = f"Failed to rank endpoint: Error - {e.response.text}"
            raise ValueError(msg) from e

        data = res.json()
        if "rankings" not in data:
            logger.error("Expected 'rankings' in response, got {d}", d=data)
            msg = f"Expected 'rankings' in response, got {data}"
            raise ValueError(msg)

        return data["rankings"]
