# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, SystemOneResponse, TypeSafeClient

_LOW_CONFIDENCE = "low_confidence"


@component
class TypeSafeTextRouter:
    """
    Routes a text to the connection of the label a TypeSafe System One model picks for it.

    The model answers a single `choice` question over `labels` and returns calibrated probabilities. The component
    has one output per label. When `min_confidence` is set, it also has a `low_confidence` output that receives the
    text whenever the answer's `confidence` is below the threshold, so uncertain texts can go to a fallback such as
    an LLM or a human. See [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing).

    The component also works with servers that implement the TypeSafe API, such as [Ollaya](https://ollaya.dev),
    which runs open decision models locally. Set `api_base_url` to the server and `model` to one of its models.

    ### Usage example

    ```python
    from haystack_integrations.components.routers.typesafe import TypeSafeTextRouter

    router = TypeSafeTextRouter(
        labels={
            "billing": "Payments, invoices, refunds and charges",
            "technical": "Bugs, crashes and errors in the product",
        },
        instructions="Which team should handle this support request?",
        min_confidence=0.6,
    )

    print(router.run(text="I was charged twice for my subscription."))
    # {'billing': 'I was charged twice for my subscription.'}
    ```
    """

    def __init__(
        self,
        labels: list[str] | dict[str, str | None],
        *,
        instructions: str = "Which category does this text belong to?",
        model: str = "jev-latest",
        api_key: Secret = Secret.from_env_var("TYPESAFE_API_KEY"),
        api_base_url: str | None = None,
        min_confidence: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """
        Creates a TypeSafeTextRouter.

        :param labels:
            The labels to route between, each becoming an output connection. Pass a dict of label to description to
            tell the model what each label means.
        :param instructions:
            The question the model answers by picking one of the labels.
        :param model:
            Name of the System One model to use.
        :param api_key:
            The TypeSafe API key. Servers such as Ollaya accept any non-empty value unless they are configured
            with a key.
        :param api_base_url:
            Base URL of the API. If `None`, the `TYPESAFE_BASE_URL` environment variable is used, or
            `https://api.typesafe.ai` if it is unset.
        :param min_confidence:
            Threshold between 0 and 1 for the answer's `confidence`. Texts below it are sent to the `low_confidence`
            output. If `None`, every text goes to a label output.
        :param timeout:
            Timeout in seconds for each request. If `None`, the SDK default of 10 seconds is used.
        :param max_retries:
            Maximum number of retries after a failed request. If `None`, the SDK default of 2 is used.
        :raises ValueError:
            If `labels` is empty or contains `low_confidence`, or if `min_confidence` is outside 0 to 1.
        """
        if not labels:
            msg = "`labels` must contain at least one label."
            raise ValueError(msg)
        if _LOW_CONFIDENCE in labels:
            msg = f"`{_LOW_CONFIDENCE}` is reserved for the low confidence output and can't be used as a label."
            raise ValueError(msg)
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            msg = f"`min_confidence` must be between 0 and 1, got {min_confidence}."
            raise ValueError(msg)

        self.labels = labels
        self.instructions = instructions
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: TypeSafeClient | None = None
        self._async_client: AsyncTypeSafeClient | None = None

        # TypeSafe takes choice criteria as a dict of label to description
        criteria = labels if isinstance(labels, dict) else dict.fromkeys(labels)
        self._question = Choice(instructions=instructions, criteria=criteria)

        outputs = list(labels)
        if min_confidence is not None:
            outputs.append(_LOW_CONFIDENCE)
        component.set_output_types(self, **dict.fromkeys(outputs, str))

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
            labels=self.labels,
            instructions=self.instructions,
            model=self.model,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            min_confidence=self.min_confidence,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TypeSafeTextRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _route(self, text: str, response: SystemOneResponse) -> dict[str, str]:
        answer = response.choices["route"]
        if self.min_confidence is not None and answer.confidence < self.min_confidence:
            return {_LOW_CONFIDENCE: text}
        return {answer.choice: text}

    def run(self, text: str) -> dict[str, str]:
        """
        Routes the text to the output of the label the model picks.

        :param text:
            The text to route.
        :returns:
            A dictionary with a single key, the picked label or `low_confidence`, mapped to the input text.
        :raises TypeError:
            If `text` is not a string.
        """
        if not isinstance(text, str):
            msg = "TypeSafeTextRouter expects a str as input."
            raise TypeError(msg)

        self.warm_up()
        assert self._client is not None  # noqa: S101  # mypy: the client is created by warm_up above

        return self._route(text, self._client.system_one(text, {"route": self._question}))

    async def run_async(self, text: str) -> dict[str, str]:
        """
        Asynchronously routes the text to the output of the label the model picks.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param text:
            The text to route.
        :returns:
            A dictionary with a single key, the picked label or `low_confidence`, mapped to the input text.
        :raises TypeError:
            If `text` is not a string.
        """
        if not isinstance(text, str):
            msg = "TypeSafeTextRouter expects a str as input."
            raise TypeError(msg)

        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101  # mypy: the client is created by warm_up_async above

        return self._route(text, await self._async_client.system_one(text, {"route": self._question}))
