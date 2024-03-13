from typing import Any, Dict, Optional

from .backend import EmbedderBackend


class NimBackend(EmbedderBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        batch_size: int,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        pass
