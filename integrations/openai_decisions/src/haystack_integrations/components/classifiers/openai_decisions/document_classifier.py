# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import mimetypes
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from haystack import Document, component, default_to_dict, logging
from haystack.dataclasses import ImageContent
from haystack.utils import Secret
from openai import OpenAI
from openai.types import DecisionInputMessageParam
from openai.types.decision_create_params import Question

logger = logging.getLogger(__name__)

_MAX_QUESTIONS = 200
_MIN_CHOICES = 2
_MAX_CHOICES = 255


def _encode_image(image: ImageContent) -> str:
    """Convert a Haystack ImageContent value to a Decisions API inline data URL."""
    payload = getattr(image, "base64_image", None)
    mime_type = getattr(image, "mime_type", None)
    if payload:
        if payload.startswith("data:"):
            return payload
        mime_type = mime_type or "image/png"
        return f"data:{mime_type};base64,{payload}"
    url = getattr(image, "url", None)
    if url:
        # This integration currently requires inline or local image data.
        msg = (
            "URL-backed ImageContent is not supported by this integration; "
            "supply base64 image data or a local file path instead."
        )
        raise ValueError(msg)
    file_path = getattr(image, "file_path", None)
    if file_path:
        path = Path(file_path)
        mime_type = mime_type or mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{payload}"
    msg = "ImageContent must contain base64 image data or a local file path."
    raise ValueError(msg)


def _document_input(content: Any) -> str | list[dict[str, Any]]:
    """Convert document text/image content into the input shape supported by the Decisions API."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                parts.append({"type": "input_text", "text": item})
            elif isinstance(item, ImageContent):
                parts.append({"type": "input_image", "image_url": _encode_image(item)})
            else:
                msg = f"Unsupported document content part: {type(item).__name__}"
                raise TypeError(msg)
        return [{"role": "user", "content": parts}]
    msg = "Document content must be a string or a list of text and ImageContent parts."
    raise TypeError(msg)


@component
class OpenAIDecisionsDocumentClassifier:
    """
    Classify Haystack documents with typed questions using the OpenAI Decisions API.

    Each document is submitted once with all configured questions. Predicate, choice, and score answers are stored
    under `meta[metadata_field]`. Text and Haystack ImageContent input parts are supported. Failed documents are
    returned separately unless `raise_on_failure=True`.
    """

    def __init__(
        self,
        questions: dict[str, dict[str, Any]],
        *,
        model: str = "gpt-6-luna",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        classification_field: str | None = None,
        metadata_field: str = "openai_decisions",
        timeout: float | None = None,
        max_retries: int | None = None,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Create an OpenAIDecisionsDocumentClassifier.

        :param questions: Questions keyed by unique answer name. Each question must define a supported type and
            instructions; choice questions require `choices`, score questions require `levels`.
        :param model: OpenAI Decisions model name.
        :param api_key: OpenAI API key, typically sourced from `OPENAI_API_KEY`.
        :param classification_field: Optional document field to classify instead of Document.content.
        :param metadata_field: Document metadata key where answers are stored.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of retries after a failed request.
        :param raise_on_failure: If true, propagate request or document conversion failures.
        :raises ValueError: If questions are empty or malformed.
        """
        if not questions:
            msg = "`questions` must contain at least one question."
            raise ValueError(msg)

        if len(questions) > _MAX_QUESTIONS:
            msg = "`questions` cannot contain more than 200 questions."
            raise ValueError(msg)

        supported = {"predicate", "choice", "score"}

        for name, question in questions.items():
            if question.get("type") not in supported:
                msg = f"Question {name!r} must have type one of {sorted(supported)}."
                raise ValueError(msg)

            if not question.get("instructions"):
                msg = f"Question {name!r} needs `instructions`."
                raise ValueError(msg)

            if question["type"] == "choice":
                choices = question.get("choices")

                if not isinstance(choices, list) or not choices:
                    msg = f"Choice question {name!r} needs `choices`."
                    raise ValueError(msg)

                if not _MIN_CHOICES <= len(choices) <= _MAX_CHOICES:
                    msg = f"Choice question {name!r} must have between 2 and 255 choices."
                    raise ValueError(msg)

                choice_values = []

                for choice in choices:
                    if not isinstance(choice, dict) or "value" not in choice:
                        msg = f"Each choice in question {name!r} must define a `value`."
                        raise ValueError(msg)

                    value = choice["value"]

                    if not isinstance(value, (str, bool)):
                        msg = f"Choice values in question {name!r} must be strings or booleans."
                        raise ValueError(msg)

                    choice_values.append((type(value), value))

                if len(set(choice_values)) != len(choice_values):
                    msg = f"Choice question {name!r} must not contain duplicate values."
                    raise ValueError(msg)

            if question["type"] == "score" and not question.get("levels"):
                msg = f"Score question {name!r} needs `levels`."
                raise ValueError(msg)

        self.questions = questions
        self.model = model
        self.api_key = api_key
        self.classification_field = classification_field
        self.metadata_field = metadata_field
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_on_failure = raise_on_failure
        self._client: OpenAI | None = None

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
            questions=self.questions,
            model=self.model,
            api_key=self.api_key,
            classification_field=self.classification_field,
            metadata_field=self.metadata_field,
            timeout=self.timeout,
            max_retries=self.max_retries,
            raise_on_failure=self.raise_on_failure,
        )

    def _input_for(self, document: Document) -> str | list[dict[str, Any]]:
        value = document.content if self.classification_field is None else document.meta.get(self.classification_field)
        if value is None:
            msg = "No content available to classify."
            raise ValueError(msg)
        return _document_input(value)

    def _classify(self, document: Document) -> Document:
        """Submit one document and return a copy carrying the typed answers."""
        assert self._client is not None  # noqa: S101
        response = self._client.decisions.create(
            model=self.model,
            input=cast(str | list[DecisionInputMessageParam], self._input_for(document)),
            questions=cast(list[Question], [{**question, "name": name} for name, question in self.questions.items()]),
        )
        answers: dict[str, Any] = {}

        for answer in response.answers:
            if answer.type == "refusal":
                msg = f"OpenAI Decisions refused question {answer.name!r}."
                raise ValueError(msg)

            name = answer.name

            if name is None:
                continue

            answers[name] = answer.model_dump(mode="json", exclude_none=True)
        meta = dict(document.meta)
        meta.pop("classification_error", None)
        meta[self.metadata_field] = answers
        return replace(document, meta=meta)

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Classify the documents, returning successes and failures separately."""
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = "`documents` must be a list of Haystack Document objects."
            raise TypeError(msg)
        if not documents:
            return {"documents": [], "failed_documents": []}
        self.warm_up()
        succeeded: list[Document] = []
        failed: list[Document] = []
        for document in documents:
            try:
                succeeded.append(self._classify(document))
            except Exception as exc:
                if self.raise_on_failure:
                    raise
                logger.warning(f"OpenAI Decisions classification failed: {exc}")
                failed.append(replace(document, meta={**document.meta, "classification_error": str(exc)}))
        return {"documents": succeeded, "failed_documents": failed}
