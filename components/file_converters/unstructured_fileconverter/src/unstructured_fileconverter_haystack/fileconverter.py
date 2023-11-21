from typing import Optional, List, Union, Literal, Dict, Any
import os
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from haystack.preview import component, Document, default_to_dict, default_from_dict
from unstructured.partition.api import partition_via_api
from unstructured.documents.elements import Element


logger = logging.getLogger(__name__)

UNSTRUCTURED_HOSTED_API_URL = "https://api.unstructured.io/general/v0/general"


@component
class UnstructuredFileConverter:
    """
    Convert files to Haystack Documents using the Unstructured API (hosted or running locally).
    """

    def __init__(
        self,
        api_url: str = UNSTRUCTURED_HOSTED_API_URL,
        api_key: Optional[str] = None,
        document_creation_mode: Literal[
            "one-doc-per-file", "one-doc-per-page", "one-doc-per-element"
        ] = "one-doc-per-file",
        separator: str = "\n\n",
        unstructured_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param api_url: URL of the Unstructured API. Defaults to the hosted version.
            If you run the API locally, specify the URL of your local API (e.g. http://localhost:8000/general/v0/general).
            See https://unstructured-io.github.io/unstructured/api.html#using-the-api-locally for more information.
        :param api_key: API key for the Unstructured API (https://unstructured.io/#get-api-key).
            If you run the API locally, it is not needed.
            If you use the hosted version, it defaults to the environment variable UNSTRUCTURED_API_KEY.
        :param document_creation_mode: How to create Haystack Documents from the elements returned by Unstructured.
            - "one-doc-per-file": One Haystack Document per file. All elements are concatenated into one text field.
            - "one-doc-per-page": One Haystack Document per page. All elements on a page are concatenated into one text field.
            - "one-doc-per-element": One Haystack Document per element. Each element is converted to a Haystack Document.
        :param separator: Separator between elements when concatenating them into one text field.
        :param unstructured_kwargs: Additional keyword arguments that are passed to the Unstructured API.
            See https://unstructured-io.github.io/unstructured/api.html.
        """

        self.api_url = api_url
        self.document_creation_mode = document_creation_mode
        self.unstructured_kwargs = unstructured_kwargs or {}
        self.separator = separator

        is_hosted_api = api_url == UNSTRUCTURED_HOSTED_API_URL
        if api_key is None and is_hosted_api:
            try:
                api_key = os.environ["UNSTRUCTURED_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "To use the hosted version of Unstructured, you need to set the environment variable "
                    "UNSTRUCTURED_API_KEY (recommended) or explictly pass the parameter api_key."
                ) from e
        self.api_key = api_key

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """

        # do not serialize api_key
        return default_to_dict(
            self,
            api_url=self.api_url,
            document_creation_mode=self.document_creation_mode,
            separator=self.separator,
            unstructured_kwargs=self.unstructured_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnstructuredFileConverter":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, paths: Union[List[str], List[os.PathLike]]):
        """
        Convert files to Haystack Documents using the Unstructured API (hosted or running locally).

        :param paths: List of paths to convert. Paths can be files or directories.
            If a path is a directory, all files in the directory are converted. Subdirectories are ignored.
        """

        paths = {Path(path) for path in paths}
        filepaths = {str(path) for path in paths if path.is_file()}
        filepaths = filepaths.union(
            {str(filepath) for path in paths if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()}
        )

        # currently, the files are converted sequentially to gently handle API failures
        documents = []

        for filepath in tqdm(filepaths):
            elements = self._partition_file_into_elements(filepath=filepath)
            docs_for_file = self._create_documents(
                filepath=filepath,
                elements=elements,
                document_creation_mode=self.document_creation_mode,
                separator=self.separator,
            )
            documents.extend(docs_for_file)

        return {"documents": documents}

    def _create_documents(
        self,
        filepath: str,
        elements: List[Element],
        document_creation_mode: Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"],
        separator: str,
    ) -> List[Document]:
        """
        Create Haystack Documents from the elements returned by Unstructured.
        """
        docs = []

        if document_creation_mode == "one-doc-per-file":
            text = separator.join([str(el) for el in elements])
            docs = [Document(content=text, meta={"name": filepath})]

        elif document_creation_mode == "one-doc-per-page":
            texts_per_page = defaultdict(str)
            meta_per_page = defaultdict(dict)
            for el in elements:
                metadata = {"name": filepath}
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                page_number = metadata.get("page_number", 1)

                texts_per_page[page_number] += str(el) + separator
                meta_per_page[page_number].update(metadata)

            docs = [Document(content=texts_per_page[page], meta=meta_per_page[page]) for page in texts_per_page.keys()]

        elif document_creation_mode == "one-doc-per-element":
            for el in elements:
                metadata = {"name": filepath}
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                if hasattr(el, "category"):
                    metadata["category"] = el.category
                doc = Document(content=str(el), meta=metadata)
                docs.append(doc)

        return docs

    def _partition_file_into_elements(self, filepath: str) -> List[Element]:
        """
        Partition a file into elements using the Unstructured API.
        """
        elements = []
        try:
            elements = partition_via_api(
                filename=filepath, api_url=self.api_url, api_key=self.api_key, **self.unstructured_kwargs
            )
        except Exception as e:
            logger.warning(f"Unstructured could not process file {filepath}. Error: {e}")
        return elements