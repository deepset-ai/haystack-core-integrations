# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from haystack import Document, component, default_to_dict, logging
from haystack.utils import Secret
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    Question,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)

logger = logging.getLogger(__name__)

_QUESTION_TYPES = ("choice", "score", "noul")
_ERROR_FIELD = "classification_error"


@component
class TypeSafeDocumentClassifier:
    """
    Answers typed questions about each document with a TypeSafe System One model and stores the answers in its metadata.

    System One models such as Jev don't generate text: they answer every question in one request and return
    calibrated probabilities. Each question has one of three types:
    - `choice`: picks one label from `criteria`, a dict of label to description (or `None`).
    - `score`: rates the text on the ordered levels in `criteria`, a list of level descriptions.
    - `noul`: returns the probability that the yes/no question in `instructions` is true.

    Answers are stored under `meta[metadata_field]`, keyed by question ID. Questions can be dicts or the SDK's
    `Choice`, `Score` and `Noul` objects. See the [TypeSafe documentation](https://docs.typesafe.ai/primitives) for the
    full question and answer format.

    Documents without text to classify are returned in `failed_documents` with the reason in
    `meta["classification_error"]`. So are documents whose request failed, unless `raise_on_failure` is `True`.

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
    print(result["failed_documents"])
    # []
    ```
    """

    def __init__(
        self,
        questions: dict[str, Question],
        *,
        model: str = "jev-latest",
        api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
        api_base_url: str | None = None,
        classification_field: str | None = None,
        metadata_field: str = "typesafe",
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates a TypeSafeDocumentClassifier.

        :param questions:
            Questions to answer for every document, keyed by question ID. Each value is a dict or an SDK `Choice`,
            `Score` or `Noul` object with a `type` (`choice`, `score` or `noul`), `instructions`, and for `choice` and
            `score` questions, `criteria`.
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
            Timeout in seconds for each request attempt. If `None`, the SDK default of 10 seconds is used.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param max_workers:
            Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed request raises its error. If `False`, documents whose request failed are
            returned in `failed_documents`.
        :raises ValueError:
            If `questions` is empty, a question has an unknown `type`, a `choice` or `score` question has no
            `criteria`, or `max_workers` is smaller than 1.
        """
        if not questions:
            msg = "`questions` must contain at least one question."
            raise ValueError(msg)
        for question_id, question in questions.items():
            # SDK question objects are validated by the SDK when they are created
            if isinstance(question, (Choice, Score, Noul)):
                continue
            if question.get("type") not in _QUESTION_TYPES:
                msg = f"Question '{question_id}' has type {question.get('type')!r}; use one of {_QUESTION_TYPES}."
                raise ValueError(msg)
            if question["type"] != "noul" and not question.get("criteria"):
                msg = f"Question '{question_id}' of type '{question['type']}' needs `criteria`."
                raise ValueError(msg)
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
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
        self.raise_on_failure = raise_on_failure
        self._client: TypeSafeClient | None = None
        self._async_client: AsyncTypeSafeClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        # The SDK's retry budget defaults to 30 seconds in total, which would skip retries after a long attempt.
        # `timeout` already bounds each attempt, so the budget is disabled.
        retry_kwargs: dict[str, Any] = {"timeout": None}
        if self.max_retries is not None:
            retry_kwargs["max_retries"] = self.max_retries
        return {
            "api_key": self.api_key.resolve_value(),
            "base_url": self.api_base_url,
            "model": self.model,
            "timeout": self.timeout,
            "retry": RetryPolicy(**retry_kwargs),
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
            questions={
                question_id: question.model_dump() if isinstance(question, (Choice, Score, Noul)) else question
                for question_id, question in self.questions.items()
            },
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            classification_field=self.classification_field,
            metadata_field=self.metadata_field,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )

    def _succeed(self, document: Document, response: SystemOneResponse) -> tuple[Document, bool]:
        """
        Adds the answers from a TypeSafe response to a copy of the document.

        :param document:
            The classified document.
        :param response:
            The TypeSafe response for the document.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and no `classification_error`, and
            `True` to mark it as classified.
        """
        # JSON mode turns the integer score levels into string keys, so the metadata stays JSON-serializable
        answers = {question_id: answer.model_dump(mode="json") for question_id, answer in response.answers.items()}
        # Drop the error left by an earlier failed run
        meta = {key: value for key, value in document.meta.items() if key != _ERROR_FIELD}
        return replace(document, meta={**meta, self.metadata_field: answers}), True

    def _classify(self, client: TypeSafeClient, document: Document) -> tuple[Document, bool]:
        """
        Sends one document to the TypeSafe API and adds the answers to a copy of it.

        :param client:
            The synchronous TypeSafe client.
        :param document:
            The document to classify.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises TypeSafeError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = document.content if self.classification_field is None else document.meta.get(self.classification_field)
        if not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            response = client.system_one(state=text, questions=self.questions)
        except TypeSafeError as error:
            if self.raise_on_failure:
                raise
            logger.warning("Classifying document {document_id} failed: {error}", document_id=document.id, error=error)
            return replace(document, meta={**document.meta, _ERROR_FIELD: str(error)}), False
        return self._succeed(document=document, response=response)

    async def _classify_async(
        self, client: AsyncTypeSafeClient, document: Document, semaphore: asyncio.Semaphore
    ) -> tuple[Document, bool]:
        """
        Asynchronously sends one document to the TypeSafe API and adds the answers to a copy of it.

        :param client:
            The asynchronous TypeSafe client.
        :param document:
            The document to classify.
        :param semaphore:
            Limits how many requests run at once.
        :returns:
            A copy of the document with the answers under `meta[metadata_field]` and `True`, or a copy with the reason
            under `meta["classification_error"]` and `False` if the document has no text or its request failed.
        :raises TypeSafeError:
            If the request fails and `raise_on_failure` is `True`.
        """
        text = document.content if self.classification_field is None else document.meta.get(self.classification_field)
        if not text:
            return replace(document, meta={**document.meta, _ERROR_FIELD: "Document has no text to classify."}), False
        try:
            async with semaphore:
                response = await client.system_one(state=text, questions=self.questions)
        except TypeSafeError as error:
            if self.raise_on_failure:
                raise
            logger.warning("Classifying document {document_id} failed: {error}", document_id=document.id, error=error)
            return replace(document, meta={**document.meta, _ERROR_FIELD: str(error)}), False
        return self._succeed(document=document, response=response)

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Answers the questions for each document and adds the answers to its metadata.

        Each document is sent as its own request, with up to `max_workers` requests running at once.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
              answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises TypeSafeError:
            If a request fails and `raise_on_failure` is `True`.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda document: self._classify(client=client, document=document), documents))
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }

    @component.output_types(documents=list[Document], failed_documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously answers the questions for each document and adds the answers to its metadata.

        This is the asynchronous version of the `run` method with the same parameters and return values. Up to
        `max_workers` requests run concurrently.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following keys:
            - `documents`: The classified documents. Each has a `metadata_field` entry mapping each question ID to its
              answer.
            - `failed_documents`: The documents that have no text to classify or whose request failed, with the
              reason in `meta["classification_error"]`.
        :raises TypeSafeError:
            If a request fails and `raise_on_failure` is `True`.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        # Bound concurrency to max_workers, mirroring the ThreadPoolExecutor in the sync `run`
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [
            asyncio.create_task(self._classify_async(client=self._async_client, document=document, semaphore=semaphore))
            for document in documents
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception:
            # We only expect to get here when `raise_on_failure` is `True`, since `_classify_async` otherwise records a
            # failed request in `failed_documents` instead of raising. Cancel the remaining requests before re-raising.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return {
            "documents": [document for document, succeeded in results if succeeded],
            "failed_documents": [document for document, succeeded in results if not succeeded],
        }
