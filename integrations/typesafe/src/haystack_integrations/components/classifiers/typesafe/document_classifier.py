# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from typesafe_sdk import AsyncTypeSafeClient, QuestionModel, RetryPolicy, SystemOneResponse, TypeSafeClient

logger = logging.getLogger(__name__)

_QUESTION_TYPES = ("choice", "score", "noul")


@component
class TypeSafeDocumentClassifier:
    """
    Answers typed questions about each document with a TypeSafe System One model and stores the answers in its metadata.

    System One models such as Jev don't generate text: they answer every question in one request and return
    calibrated probabilities. Each question has one of three types:
    - `choice`: picks one label from `criteria`, a dict of label to description (or `None`).
    - `score`: rates the text on the ordered levels in `criteria`, a list of level descriptions.
    - `noul`: returns the probability that the yes/no question in `instructions` is true.

    Answers are stored under `meta[metadata_field]`, keyed by question ID. See the
    [TypeSafe documentation](https://docs.typesafe.ai/primitives) for the full question and answer format.

    The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev),
    which runs open decision models locally. Set `api_base_url` to the server and `model` to one of its models.

    ### Usage example

    ```python
    from haystack import Document

    from haystack_integrations.components.classifiers.typesafe import TypeSafeDocumentClassifier

    classifier = TypeSafeDocumentClassifier(
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {"billing": None, "technical": None, "sales": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"],
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        }
    )

    result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
    print(result["documents"][0].meta["typesafe"])
    # {'department': {'type': 'choice', 'choice': 'billing', 'confidence': 0.9516,
    #                 'probabilities': {'billing': 0.9677, 'technical': 0.0221, 'sales': 0.0102}},
    #  'urgency': {'type': 'score', 'score': 1.948, 'confidence': 0.9389,
    #              'legend': {'0': 'can wait', '1': 'this week', '2': 'today'},
    #              'probabilities': {'0': 0.0113, '1': 0.0295, '2': 0.9593}},
    #  'refund': {'type': 'noul', 'noul': 0.9498}}
    ```
    """

    def __init__(
        self,
        questions: dict[str, QuestionModel],
        *,
        model: str = "jev-latest",
        api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
        api_base_url: str | None = None,
        classification_field: str | None = None,
        metadata_field: str = "typesafe",
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
    ) -> None:
        """
        Creates a TypeSafeDocumentClassifier.

        :param questions:
            Questions to answer for every document, keyed by question ID. Each value is a dict with a `type`
            (`choice`, `score` or `noul`), `instructions`, and for `choice` and `score` questions, `criteria`.
        :param model:
            Name of the System One model to use.
        :param api_key:
            The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
            with a key.
        :param api_base_url:
            Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
            `https://api.typesafe.ai` if it is unset.
        :param classification_field:
            Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
        :param metadata_field:
            Name of the metadata field the answers are written to.
        :param timeout:
            Timeout in seconds for each request. If `None`, the SDK default of 10 seconds is used.
        :param max_retries:
            Maximum number of retries after a failed request. If `None`, the SDK default of 2 is used.
        :param max_workers:
            Maximum number of worker threads in `run` and of concurrent requests in `run_async`.
        :raises ValueError:
            If `questions` is empty, or a question has an unknown `type`, or a `choice` or `score` question has no
            `criteria`.
        """
        if not questions:
            msg = "`questions` must contain at least one question."
            raise ValueError(msg)
        for question_id, question in questions.items():
            if question.get("type") not in _QUESTION_TYPES:
                msg = f"Question '{question_id}' has type {question.get('type')!r}; use one of {_QUESTION_TYPES}."
                raise ValueError(msg)
            if question["type"] != "noul" and not question.get("criteria"):
                msg = f"Question '{question_id}' of type '{question['type']}' needs `criteria`."
                raise ValueError(msg)

        self.questions = questions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.classification_field = classification_field
        self.metadata_field = metadata_field
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self._client: TypeSafeClient | None = None
        self._async_client: AsyncTypeSafeClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        return {
            "api_key": self.api_key.resolve_value(),
            "base_url": self.api_base_url,
            "model": self.model,
            "timeout": self.timeout,
            "retry": RetryPolicy(max_retries=self.max_retries) if self.max_retries is not None else None,
        }

    def warm_up(self) -> None:
        """
        Creates the synchronous TypeSafe client.
        """
        if self._client is None:
            self._client = TypeSafeClient(**self._client_kwargs())

    async def warm_up_async(self) -> None:
        """
        Creates the asynchronous TypeSafe client.
        """
        if self._async_client is None:
            self._async_client = AsyncTypeSafeClient(**self._client_kwargs())

    def close(self) -> None:
        """
        Releases the synchronous TypeSafe client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Releases the asynchronous TypeSafe client.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            questions=self.questions,
            model=self.model,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            classification_field=self.classification_field,
            metadata_field=self.metadata_field,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypeSafeDocumentClassifier":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _texts_to_classify(self, documents: list[Document]) -> dict[int, str]:
        # Map each document's position to its text, skipping documents that have none
        texts = {}
        for index, document in enumerate(documents):
            text = (
                document.content if self.classification_field is None else document.meta.get(self.classification_field)
            )
            if not text:
                logger.warning(
                    "Document {document_id} has no text to classify and is skipped.", document_id=document.id
                )
                continue
            texts[index] = text
        return texts

    def _add_answers(self, documents: list[Document], responses: dict[int, SystemOneResponse]) -> list[Document]:
        new_documents = list(documents)
        for index, response in responses.items():
            document = documents[index]
            # JSON mode turns the integer score levels into string keys, so the metadata stays JSON-serializable
            answers = {question_id: answer.model_dump(mode="json") for question_id, answer in response.answers.items()}
            new_documents[index] = replace(document, meta={**document.meta, self.metadata_field: answers})
        return new_documents

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Answers the questions for each document and adds the answers to its metadata.

        Each document is sent as its own request, with up to `max_workers` requests running at once. Documents
        without text to classify are returned unchanged and a warning is logged.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following key:
            - `documents`: The input documents. Each classified document has a `metadata_field` entry mapping each
              question ID to its answer.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        texts = self._texts_to_classify(documents)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(lambda text: client.system_one(text, self.questions), texts.values())
            responses = dict(zip(texts, results, strict=True))
        return {"documents": self._add_answers(documents, responses)}

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously answers the questions for each document and adds the answers to its metadata.

        This is the asynchronous version of the `run` method with the same parameters and return values. Up to
        `max_workers` requests run concurrently.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following key:
            - `documents`: The input documents. Each classified document has a `metadata_field` entry mapping each
              question ID to its answer.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        client = self._async_client
        texts = self._texts_to_classify(documents)

        # Bound concurrency to max_workers, mirroring the ThreadPoolExecutor in the sync `run`
        semaphore = asyncio.Semaphore(max(1, self.max_workers))

        async def _bounded_system_one(text: str) -> SystemOneResponse:
            async with semaphore:
                return await client.system_one(text, self.questions)

        results = await asyncio.gather(*(_bounded_system_one(text) for text in texts.values()))
        responses = dict(zip(texts, results, strict=True))
        return {"documents": self._add_answers(documents, responses)}
