from abc import ABC, abstractmethod
from typing import Dict, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast


class DefaultPromptHandler:
    """
    DefaultPromptHandler resizes the prompt to ensure that the prompt and answer token lengths together
    are within the model_max_length.
    """

    def __init__(self, tokenizer: Union[str, PreTrainedTokenizerBase], model_max_length: int, max_length: int = 100):
        """
        :param tokenizer: The tokenizer to be used to tokenize the prompt.
        :param model_max_length: The maximum length of the prompt and answer tokens combined.
        :param max_length: The maximum length of the answer tokens.
        :raises ValueError: If the tokenizer is not a string or a `PreTrainedTokenizer` or `PreTrainedTokenizerFast`
            instance.
        """
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer = tokenizer
        else:
            msg = "model must be a string or a PreTrainedTokenizer instance"
            raise ValueError(msg)

        self.tokenizer.model_max_length = model_max_length
        self.model_max_length = model_max_length
        self.max_length = max_length

    def __call__(self, prompt: str, **kwargs) -> Dict[str, Union[str, int]]:
        """
        Resizes the prompt to ensure that the prompt and answer is within the model_max_length

        :param prompt: the prompt to be sent to the model.
        :param kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary containing the resized prompt and additional information.
        """
        resized_prompt = prompt
        prompt_length = 0
        new_prompt_length = 0

        if prompt:
            tokenized_prompt = self.tokenizer.tokenize(prompt)
            prompt_length = len(tokenized_prompt)
            if (prompt_length + self.max_length) <= self.model_max_length:
                resized_prompt = prompt
                new_prompt_length = prompt_length
            else:
                resized_prompt = self.tokenizer.convert_tokens_to_string(
                    tokenized_prompt[: self.model_max_length - self.max_length]
                )
                new_prompt_length = len(tokenized_prompt[: self.model_max_length - self.max_length])

        return {
            "resized_prompt": resized_prompt,
            "prompt_length": prompt_length,
            "new_prompt_length": new_prompt_length,
            "model_max_length": self.model_max_length,
            "max_length": self.max_length,
        }


class TokenStreamingHandler(ABC):
    """
    TokenStreamingHandler implementations handle the streaming of tokens from the stream.
    """

    DONE_MARKER = "[DONE]"

    @abstractmethod
    def __call__(self, token_received: str, **kwargs) -> str:
        """
        This callback method is called when a new token is received from the stream.

        :param token_received: The token received from the stream.
        :param kwargs: Additional keyword arguments passed to the handler.
        :returns: The token to be sent to the stream.
        """
        pass


class DefaultTokenStreamingHandler(TokenStreamingHandler):
    def __call__(self, token_received, **kwargs) -> str:
        """
        This callback method is called when a new token is received from the stream.

        :param token_received: The token received from the stream.
        :param kwargs: Additional keyword arguments passed to the handler.
        :returns: The token to be sent to the stream.
        """
        print(token_received, flush=True, end="")  # noqa: T201
        return token_received
