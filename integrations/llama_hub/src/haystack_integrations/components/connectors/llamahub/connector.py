import importlib
from typing import Any, Dict, List, Optional
from haystack import Document, component


@component
class LlamaHubConnector:
    """
    A unified components wrapper enabling the execution of modern, explicitly installed
    LlamaIndex readers directly within native Haystack 2.0 pipelines.
    """

    def __init__(
        self,
        reader_module: str,
        reader_class: str,
        reader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param reader_module: The specific sub-package name (e.g., 'llama_index.readers.web').
        :param reader_class: The class name inside that module (e.g., 'SimpleWebPageReader').
        :param reader_kwargs: Initialization arguments to pass to the target reader.
        """
        if not reader_module or not reader_class:
            raise ValueError(
                "Parameters 'reader_module' and 'reader_class' cannot be empty."
            )

        self.reader_module = reader_module
        self.reader_class = reader_class
        self.reader_kwargs = reader_kwargs or {}

        # Dynamically import the explicitly installed reader package
        try:
            module = importlib.import_module(self.reader_module)
            self._loader_class = getattr(module, self.reader_class)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load '{self.reader_class}' from '{self.reader_module}'. "
                f"Ensure you have run: pip install {self.reader_module.replace('_', '-')}"
            ) from e

        self._loader_instance = self._loader_class(**self.reader_kwargs)

    @component.output_types(documents=List[Document])
    def run(self, **kwargs: Any) -> Dict[str, List[Document]]:
        # Invoke data extraction from the loaded LlamaIndex reader instance
        llama_docs = self._loader_instance.load_data(**kwargs)

        haystack_docs: List[Document] = []
        for doc in llama_docs:
            text_content = getattr(doc, "text", "") or getattr(doc, "content", "")
            meta_payload = getattr(doc, "metadata", {}) or {}

            haystack_docs.append(
                Document(content=text_content, meta=dict(meta_payload))
            )

        return {"documents": haystack_docs}
