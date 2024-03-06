# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    content: str
    role: str


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
