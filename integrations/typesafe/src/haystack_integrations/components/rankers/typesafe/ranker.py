# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from haystack import Document, component, default_to_dict, logging
from haystack.utils import Secret
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)

logger = logging.getLogger(__name__)

_QUESTION_ID = "relevance"
_ERROR_FIELD = "ranking_error"

DEFAULT_INSTRUCTIONS = "How useful is `passage` for answering `query`? Judge only that passage."
DEFAULT_CRITERIA = [
    "The passage is about something else and does not help with the query.",
    "The passage shares a keyword with the query but is about a different topic.",
    "The passage is on the right topic but only gives background; the answer is not in it.",
    "The passage answers part of the query, or answers it only indirectly.",
    "The passage directly answers the query.",
]


@dataclass
class _Ranked:
    document: Document
    relevance: float | None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@component
class TypeSafeRanker:
    """
    Ranks documents by relevance to a query with a TypeSafe System One model such as Jev.

    Each document is judged on its own with a `score` question, in one request per document. The model sees the query
    and the passage as a JSON object, `{"query": ..., "passage": ...}`, so `instructions` and `criteria` can refer to
    both. The score is the model's expected level, divided by the highest level, so it is a relevance from 0 to 1.

    The ranker also reports what the requests cost. Its `meta` output holds the model and the summed `usage`, in the
    same shape generators use, so that tools reading usage from pipeline traces can pick it up.

    Documents whose request failed are kept after every scored document, with the reason in `meta["ranking_error"]`,
    unless `raise_on_failure` is `True`. They are only part of the output if fewer than `top_k` documents were scored.

    The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev).
    Set `api_base_url` to the server and `model` to one of its models.

    ### Usage example

    ```python
    from haystack import Document

    from haystack_integrations.components.rankers.typesafe import TypeSafeRanker

    ranker = TypeSafeRanker(top_k=2)
    result = ranker.run(
        query="How do I reset my password?",
        documents=[
            Document(content="Click 'Forgot password' on the login page to reset it."),
            Document(content="Our offices are closed on public holidays."),
        ],
    )
    print(result["documents"][0].content)
    # Click 'Forgot password' on the login page to reset it.
    print(result["meta"])
    # {'model': 'jev-latest', 'usage': {'input_tokens': 96, 'output_tokens': 0}}
    ```
    """

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
        api_base_url: str | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        criteria: list[str] | None = None,
        meta_fields_to_include: list[str] | None = None,
        max_chars: int | None = None,
        metadata_field: str = "typesafe",
        top_k: int = 10,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_workers: int = 3,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates a TypeSafeRanker.

        :param model:
            Name of the System One model to use.
        :param api_key:
            The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
            with a key.
        :param api_base_url:
            Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
            `https://api.typesafe.ai` if it is unset.
        :param instructions:
            The question the model answers about each document. It can refer to the `query` and the `passage`.
        :param criteria:
            The ordered score levels, from the least to the most relevant, each described by a concrete situation. At
            least two. Levels are judged independently and never see their index. If `None`, five generic levels are
            used.
        :param meta_fields_to_include:
            Names of metadata fields to show the model next to the text, under their own names in the passage. Fields a
            document doesn't have are left out.
        :param max_chars:
            Truncate each document's text to this many characters. If `None`, the text is sent in full.
        :param metadata_field:
            Name of the metadata field the raw answer is written to, under the `relevance` key. Answers already in
            this field, such as those of a `TypeSafeDocumentClassifier` using the same field, are kept.
        :param top_k:
            Maximum number of documents to return. Can be overridden in `run`.
        :param timeout:
            Timeout in seconds for each request attempt. If `None`, the SDK default of 10 seconds is used.
        :param max_retries:
            Maximum number of retries after a failed request attempt. If `None`, the SDK default of 2 is used.
        :param max_workers:
            Maximum number of worker threads in `run` and of concurrent requests in `run_async`. Must be at least 1.
        :param raise_on_failure:
            If `True`, the first failed request raises its error. If `False`, documents whose request failed are
            ranked last.
        :raises ValueError:
            If `criteria` has fewer than two levels, `top_k` is smaller than 1, `max_chars` is smaller than 1, or
            `max_workers` is smaller than 1.
        """
        criteria = list(DEFAULT_CRITERIA if criteria is None else criteria)
        if len(criteria) < 2:  # noqa: PLR2004
            msg = "`criteria` must contain at least two score levels."
            raise ValueError(msg)
        if top_k < 1:
            msg = f"`top_k` must be at least 1, got {top_k}."
            raise ValueError(msg)
        if max_chars is not None and max_chars < 1:
            msg = f"`max_chars` must be at least 1, got {max_chars}."
            raise ValueError(msg)
        if max_workers < 1:
            msg = f"`max_workers` must be at least 1, got {max_workers}."
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.instructions = instructions
        self.criteria = criteria
        self.meta_fields_to_include = meta_fields_to_include
        self.max_chars = max_chars
        self.metadata_field = metadata_field
        self.top_k = top_k
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.raise_on_failure = raise_on_failure
        self._questions = {_QUESTION_ID: Score(instructions=instructions, criteria=criteria)}
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
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            instructions=self.instructions,
            criteria=self.criteria,
            meta_fields_to_include=self.meta_fields_to_include,
            max_chars=self.max_chars,
            metadata_field=self.metadata_field,
            top_k=self.top_k,
            timeout=self.timeout,
            max_retries=self.max_retries,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )

    def _state(self, query: str, document: Document) -> dict[str, Any]:
        text = document.content or ""
        passage: dict[str, Any] = {
            name: document.meta[name] for name in self.meta_fields_to_include or [] if name in document.meta
        }
        passage["text"] = text[: self.max_chars] if self.max_chars is not None else text
        return {"query": query, "passage": passage}

    def _succeed(self, document: Document, response: SystemOneResponse) -> _Ranked:
        answer = response.answers[_QUESTION_ID]
        relevance = answer.score / (len(self.criteria) - 1)  # type: ignore[union-attr]
        usage = response.usage
        answers = document.meta.get(self.metadata_field)
        if not isinstance(answers, dict):
            if answers is not None:
                logger.warning(
                    "Overwriting the non-dict value of metadata field {field} of document {document_id}",
                    field=self.metadata_field,
                    document_id=document.id,
                )
            answers = {}
        return _Ranked(
            document=replace(
                document,
                score=relevance,
                meta={
                    **{key: value for key, value in document.meta.items() if key != _ERROR_FIELD},
                    self.metadata_field: {
                        **answers,
                        _QUESTION_ID: {"score": answer.score, "confidence": answer.confidence},  # type: ignore[union-attr]
                    },
                },
            ),
            relevance=relevance,
            model=response.model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
        )

    def _fail(self, document: Document, error: TypeSafeError) -> _Ranked:
        logger.warning("Ranking document {document_id} failed: {error}", document_id=document.id, error=error)
        return _Ranked(document=replace(document, meta={**document.meta, _ERROR_FIELD: str(error)}), relevance=None)

    def _rank_one(self, client: TypeSafeClient, query: str, document: Document) -> _Ranked:
        try:
            response = client.system_one(state=self._state(query, document), questions=self._questions)
        except TypeSafeError as error:
            if self.raise_on_failure:
                raise
            return self._fail(document, error)
        return self._succeed(document, response)

    async def _rank_one_async(
        self, client: AsyncTypeSafeClient, query: str, document: Document, semaphore: asyncio.Semaphore
    ) -> _Ranked:
        try:
            async with semaphore:
                response = await client.system_one(state=self._state(query, document), questions=self._questions)
        except TypeSafeError as error:
            if self.raise_on_failure:
                raise
            return self._fail(document, error)
        return self._succeed(document, response)

    def _result(self, ranked: list[_Ranked], top_k: int) -> dict[str, Any]:
        # `sorted` is stable, so documents with equal relevance keep their input order, and failed ones come last
        ordered = sorted(ranked, key=lambda item: -1.0 if item.relevance is None else item.relevance, reverse=True)
        models = [item.model for item in ranked if item.model]
        return {
            "documents": [item.document for item in ordered[:top_k]],
            "meta": {
                "model": models[0] if models else self.model,
                "usage": {
                    "input_tokens": sum(item.input_tokens for item in ranked),
                    "output_tokens": sum(item.output_tokens for item in ranked),
                },
            },
        }

    def _top_k(self, top_k: int | None) -> int:
        top_k = self.top_k if top_k is None else top_k
        if top_k < 1:
            msg = f"`top_k` must be at least 1, got {top_k}."
            raise ValueError(msg)
        return top_k

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, Any]:
        """
        Ranks the documents by their relevance to the query.

        Each document is sent as its own request, with up to `max_workers` requests running at once.

        :param query:
            The query to rank the documents against.
        :param documents:
            Documents to rank.
        :param top_k:
            Maximum number of documents to return. If `None`, the `top_k` set at initialization is used.
        :returns:
            A dictionary with the following keys:
            - `documents`: Up to `top_k` documents, most relevant first. Each has its relevance from 0 to 1 as `score`
              and the raw answer under `meta[metadata_field]["relevance"]`.
            - `meta`: The `model` and the `usage` (`input_tokens` and `output_tokens`) summed over all requests.
        :raises ValueError:
            If `top_k` is smaller than 1.
        :raises TypeSafeError:
            If a request fails and `raise_on_failure` is `True`.
        """
        top_k = self._top_k(top_k)
        if not documents:
            return self._result([], top_k)

        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        client = self._client
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(documents))) as executor:
            ranked = list(executor.map(lambda document: self._rank_one(client, query, document), documents))
        return self._result(ranked, top_k)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, Any]:
        """
        Asynchronously ranks the documents by their relevance to the query.

        This is the asynchronous version of the `run` method with the same parameters and return values. Up to
        `max_workers` requests run concurrently.

        :param query:
            The query to rank the documents against.
        :param documents:
            Documents to rank.
        :param top_k:
            Maximum number of documents to return. If `None`, the `top_k` set at initialization is used.
        :returns:
            A dictionary with the `documents` and `meta` keys, as in `run`.
        :raises ValueError:
            If `top_k` is smaller than 1.
        :raises TypeSafeError:
            If a request fails and `raise_on_failure` is `True`.
        """
        top_k = self._top_k(top_k)
        if not documents:
            return self._result([], top_k)

        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [
            asyncio.create_task(self._rank_one_async(self._async_client, query, document, semaphore))
            for document in documents
        ]
        try:
            ranked = await asyncio.gather(*tasks)
        except Exception:
            # We only get here when `raise_on_failure` is `True`. Cancel the remaining requests before re-raising.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return self._result(list(ranked), top_k)
