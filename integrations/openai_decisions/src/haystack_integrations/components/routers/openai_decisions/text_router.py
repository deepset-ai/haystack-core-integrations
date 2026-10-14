# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from haystack import component, default_to_dict
from haystack.utils import Secret
from openai import OpenAI
from openai.types.decision_create_params import Question

_LOW_CONFIDENCE = "low_confidence"
_MIN_LABELS = 2
_MAX_LABELS = 255


@component
class OpenAIDecisionsTextRouter:
    """
    Route text to a label selected by the OpenAI Decisions API.

    Each label becomes an output socket. Setting `min_confidence` adds a `low_confidence` fallback socket.
    """

    def __init__(
        self,
        labels: list[str] | dict[str, str | None],
        *,
        instructions: str = "Which category does this text belong to?",
        model: str = "gpt-6-luna",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Create a text router.

        :param labels: Labels to route between; a mapping may provide descriptions for choices.
        :param instructions: Question to ask the Decisions API.
        :param model: OpenAI Decisions model name.
        :param api_key: OpenAI API key.
        :param min_confidence: Optional threshold for routing to the `low_confidence` output.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of request retries.
        :raises ValueError: If labels or confidence threshold are invalid.
        """
        label_values = list(labels)

        if not _MIN_LABELS <= len(label_values) <= _MAX_LABELS:
            msg = "`labels` must contain between 2 and 255 labels."
            raise ValueError(msg)

        if len(set(label_values)) != len(label_values):
            msg = "`labels` must not contain duplicate labels."
            raise ValueError(msg)
        if _LOW_CONFIDENCE in labels:
            msg = "`low_confidence` is reserved for the fallback output."
            raise ValueError(msg)
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            msg = "`min_confidence` must be between 0 and 1."
            raise ValueError(msg)
        self.labels = labels
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: OpenAI | None = None
        criteria = labels if isinstance(labels, dict) else dict.fromkeys(labels)
        self._question = {
            "type": "choice",
            "name": "route",
            "instructions": instructions,
            "choices": [
                {"value": label, **({"description": description} if description else {})}
                for label, description in criteria.items()
            ],
        }
        outputs = list(labels)
        if min_confidence is not None:
            outputs.append(_LOW_CONFIDENCE)
        component.set_output_types(self, **dict.fromkeys(outputs, str))

    def warm_up(self) -> None:
        """Create the OpenAI client on first use."""
        if self._client is None:
            kwargs: dict[str, Any] = {"api_key": self.api_key.resolve_value(), "timeout": self.timeout}
            if self.max_retries is not None:
                kwargs["max_retries"] = self.max_retries
            self._client = OpenAI(**kwargs)

    def close(self) -> None:
        """Close the OpenAI client if it has been created."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component and its initialization parameters."""
        return default_to_dict(
            self,
            labels=self.labels,
            instructions=self.instructions,
            model=self.model,
            api_key=self.api_key,
            min_confidence=self.min_confidence,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _route(self, text: str, decision: Any) -> dict[str, str]:
        answer = next((item for item in decision.answers if item.name == "route"), None)
        if answer is None:
            msg = "The Decisions API response did not contain the 'route' answer."
            raise ValueError(msg)
        if answer.type == "refusal":
            msg = "The Decisions API refused to answer the routing question."
            raise ValueError(msg)
        if answer.type != "choice":
            msg = f"Expected a choice answer for routing, got {answer.type!r}."
            raise ValueError(msg)
        choice = answer.choice

        if not isinstance(choice, str) or choice not in self.labels:
            msg = f"Unexpected routing choice: {choice!r}"
            raise ValueError(msg)

        if self.min_confidence is not None and (answer.confidence is None or answer.confidence < self.min_confidence):
            return {_LOW_CONFIDENCE: text}

        return {choice: text}

    def run(self, text: str) -> dict[str, str]:
        """Route text to the selected label output."""
        if not isinstance(text, str):
            msg = "OpenAIDecisionsTextRouter expects a string as input."
            raise TypeError(msg)
        self.warm_up()
        assert self._client is not None  # noqa: S101
        decision = self._client.decisions.create(
            model=self.model,
            input=text,
            questions=[cast(Question, self._question)],
        )
        return self._route(text, decision)
