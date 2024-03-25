from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


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
