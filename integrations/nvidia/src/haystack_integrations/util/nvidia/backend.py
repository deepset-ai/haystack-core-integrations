from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Model:
    """
    Model information.

    id: unique identifier for the model, passed as model parameter for requests
    aliases: list of aliases for the model
    base_model: root model for the model
    All aliases are deprecated and will trigger a warning when used.
    """

    id: str
    aliases: Optional[List[str]] = field(default_factory=list)
    base_model: Optional[str] = None


class EmbedderBackend(ABC):
    def __init__(self, model: str, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the backend.

        :param model:
            The name of the model to use.
        :param model_kwargs:
            Additional keyword arguments to pass to the model.
        """
        self.model_name = model
        self.model_kwargs = model_kwargs or {}

    @abstractmethod
    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Invoke the backend and embed the given texts.

        :param texts:
            Texts to embed.
        :return:
            Vector representation of the texts and
            metadata returned by the service.
        """
        pass

    @abstractmethod
    def models(self) -> List[Model]:
        """
        Invoke the backend to get available models.

        :return:
            Available models
        """
        pass


class GeneratorBackend(ABC):
    def __init__(self, model: str, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize the backend.

        :param model:
            The name of the model to use.
        :param model_kwargs:
            Additional keyword arguments to pass to the model.
        """
        self.model_name = model
        self.model_kwargs = model_kwargs or {}

    @abstractmethod
    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Invoke the backend and prompt the model.

        :param prompt:
            Prompt text.
        :return:
            Vector representation of the generated texts related
            metadata returned by the service.
        """
        pass

    @abstractmethod
    def models(self) -> List[Model]:
        """
        Invoke the backend to get available models.

        :return:
            Available models
        """
        pass
