# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from ._schema import GenerationResponse

REQUESTS_TIMEOUT = 30

INVOKE_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions"
STATUS_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"


class ModelProvider(ABC):
    @abstractmethod
    def send(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass


@dataclass
class NvidiaProvider(ModelProvider):
    session: requests.Session
    model_id: str
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1024
    seed: Optional[int] = None
    bad: Optional[List[str]] = None
    stop: Optional[List[str]] = None

    def send(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "bad": self.bad,
            "stop": self.stop,
            "stream": False,
        }
        res = self.session.post(
            url=f"{INVOKE_ENDPOINT}/{self.model_id}",
            json=payload,
            timeout=REQUESTS_TIMEOUT,
        )
        while res.status_code == 202:
            request_id = res.headers.get("NVCF-REQID")
            res = self.session.get(
                url=f"{STATUS_ENDPOINT}/{request_id}",
                timeout=REQUESTS_TIMEOUT,
            )
        res.raise_for_status()

        replies = []
        meta = []
        data = GenerationResponse.from_dict(res.json())
        usage = data.usage
        for choice in data.choices:
            replies.append(choice.message.content)
            meta.append(
                {
                    "role": choice.message.role,
                    "finish_reason": choice.finish_reason,
                    # The usage field is not part of each choice, so we use reuse it each time
                    "completion_tokens": usage.completion_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )
        return {"replies": replies, "meta": meta}
