from typing import Any, Dict, List, Optional, Tuple

from .backend import GeneratorBackend


class NimBackend(GeneratorBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        pass

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        return [], [], {}
