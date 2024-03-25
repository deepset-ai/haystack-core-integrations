# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Callable, Dict, List, Optional, cast

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import deserialize_callback_handler, serialize_callback_handler
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_secrets_inplace

from cohere import COHERE_API_URL, Client
from cohere.responses import Generations

logger = logging.getLogger(__name__)


@component
class CohereGenerator:
    """LLM Generator compatible with Cohere's generate endpoint.

    Queries the LLM using Cohere's API. Invocations are made using 'cohere' package.
    See [Cohere API](https://docs.cohere.com/reference/generate) for more details.

    Example usage:

    ```python
    from haystack_integrations.components.generators.cohere import CohereGenerator

    generator = CohereGenerator(api_key="test-api-key")
    generator.run(prompt="What's the capital of France?")
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "command",
        streaming_callback: Optional[Callable] = None,
        api_base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Instantiates a `CohereGenerator` component.

        :param api_key: the API key for the Cohere API.
        :param model: the name of the model to use. Available models are: [command, command-light, command-nightly,
            command-nightly-light].
        :param streaming_callback: A callback function to be called with the streaming response.
        :param api_base_url: the base URL of the Cohere API.
        :param kwargs: additional model parameters. These will be used during generation. Refer to
            https://docs.cohere.com/reference/generate for more details.
            Some of the parameters are:
            - 'max_tokens': The maximum number of tokens to be generated. Defaults to 1024.
            - 'truncate': One of NONE|START|END to specify how the API will handle inputs longer than the maximum token
                length. Defaults to END.
            - 'temperature': A non-negative float that tunes the degree of randomness in generation. Lower temperatures
                mean less random generations.
            - 'preset': Identifier of a custom preset. A preset is a combination of parameters, such as prompt,
                temperature etc. You can create presets in the playground.
            - 'end_sequences': The generated text will be cut at the beginning of the earliest occurrence of an end
                sequence. The sequence will be excluded from the text.
            - 'stop_sequences': The generated text will be cut at the end of the earliest occurrence of a stop sequence.
                The sequence will be included the text.
            - 'k': Defaults to 0, min value of 0.01, max value of 0.99.
            - 'p': Ensures that only the most likely tokens, with total probability mass of `p`, are considered for
                generation at each step. If both `k` and `p` are enabled, `p` acts after `k`.
            - 'frequency_penalty': Used to reduce repetitiveness of generated tokens. The higher the value, the stronger
                a penalty is applied to previously present tokens, proportional to how many times they have already
                appeared in the prompt or prior generation.'
            - 'presence_penalty': Defaults to 0.0, min value of 0.0, max value of 1.0. Can be used to reduce
                repetitiveness of generated tokens. Similar to `frequency_penalty`, except that this penalty is applied
                equally to all tokens that have already appeared, regardless of their exact frequencies.
            - 'return_likelihoods': One of GENERATION|ALL|NONE to specify how and if the token likelihoods are returned
                with the response. Defaults to NONE.
            - 'logit_bias': Used to prevent the model from generating unwanted tokens or to incentivize it to include
                desired tokens. The format is {token_id: bias} where bias is a float between -10 and 10.
        """
        if not api_base_url:
            api_base_url = COHERE_API_URL

        self.api_key = api_key
        self.model = model
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.model_parameters = kwargs
        self.client = Client(api_key=self.api_key.resolve_value(), api_url=self.api_base_url, client_name="haystack")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None,
            api_base_url=self.api_base_url,
            api_key=self.api_key.to_dict(),
            **self.model_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
           Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        if "streaming_callback" in init_params and init_params["streaming_callback"] is not None:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(
                init_params["streaming_callback"]
            )
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompt: the prompt to be sent to the generative model.
        :returns: A dictionary with the following keys:
            - `replies`: the list of replies generated by the model.
            - `meta`: metadata about the request.
        """
        response = self.client.generate(
            model=self.model, prompt=prompt, stream=self.streaming_callback is not None, **self.model_parameters
        )
        if self.streaming_callback:
            metadata_dict: Dict[str, Any] = {}
            for chunk in response:
                stream_chunk = self._build_chunk(chunk)
                self.streaming_callback(stream_chunk)
            replies = response.texts
            metadata_dict["finish_reason"] = response.finish_reason
            metadata = [metadata_dict]
            self._check_truncated_answers(metadata)
            return {"replies": replies, "meta": metadata}

        metadata = [{"finish_reason": resp.finish_reason} for resp in cast(Generations, response)]
        replies = [resp.text for resp in response]
        self._check_truncated_answers(metadata)
        return {"replies": replies, "meta": metadata}

    def _build_chunk(self, chunk) -> StreamingChunk:
        """
        Converts the response from the Cohere API to a StreamingChunk.
        :param chunk: The chunk returned by the OpenAI API.
        :returns: The StreamingChunk.
        """
        streaming_chunk = StreamingChunk(content=chunk.text, meta={"index": chunk.index})
        return streaming_chunk

    def _check_truncated_answers(self, metadata: List[Dict[str, Any]]):
        """
        Check the `finish_reason` returned with the Cohere response.
        If the `finish_reason` is `MAX_TOKEN`, log a warning to the user.
        :param metadata: The metadata returned by the Cohere API.
        """
        if metadata[0]["finish_reason"] == "MAX_TOKENS":
            logger.warning(
                "Responses have been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions."
            )
