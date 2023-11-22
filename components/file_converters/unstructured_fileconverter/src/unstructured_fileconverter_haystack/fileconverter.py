import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack.preview import Document, component, default_to_dict
from tqdm import tqdm
from unstructured.documents.elements import Element  # type: ignore[import]
from unstructured.partition.api import partition_via_api  # type: ignore[import]

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
        progress_bar: bool = True,  # noqa: FBT001, FBT002
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
            - "one-doc-per-page": One Haystack Document per page.
               All elements on a page are concatenated into one text field.
            - "one-doc-per-element": One Haystack Document per element.
              Each element is converted to a Haystack Document.
        :param separator: Separator between elements when concatenating them into one text field.
        :param unstructured_kwargs: Additional keyword arguments that are passed to the Unstructured API.
            See https://unstructured-io.github.io/unstructured/api.html.
        :param progress_bar: Show a progress bar for the conversion. Defaults to True.
        """

        self.api_url = api_url
        self.document_creation_mode = document_creation_mode
        self.unstructured_kwargs = unstructured_kwargs or {}
        self.separator = separator
        self.progress_bar = progress_bar

        is_hosted_api = api_url == UNSTRUCTURED_HOSTED_API_URL
        if api_key is None and is_hosted_api:
            try:
                api_key = os.environ["UNSTRUCTURED_API_KEY"]
            except KeyError as e:
                msg = (
                    "To use the hosted version of Unstructured, you need to set the environment variable "
                    "UNSTRUCTURED_API_KEY (recommended) or explictly pass the parameter api_key."
                )
                raise ValueError(msg) from e
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
            progress_bar=self.progress_bar,
        )

    @component.output_types(documents=List[Document])
    def run(self, paths: Union[List[str], List[os.PathLike]]):
        """
        Convert files to Haystack Documents using the Unstructured API (hosted or running locally).

        :param paths: List of paths to convert. Paths can be files or directories.
            If a path is a directory, all files in the directory are converted. Subdirectories are ignored.
        """

        unique_paths = {Path(path) for path in paths}
        filepaths = {path for path in unique_paths if path.is_file()}
        filepaths_in_directories = {
            filepath for path in unique_paths if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()
        }

        all_filepaths = filepaths.union(filepaths_in_directories)

        # currently, the files are converted sequentially to gently handle API failures
        documents = []

        for filepath in tqdm(
            all_filepaths, desc="Converting files to Haystack Documents", disable=not self.progress_bar
        ):
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
        filepath: Path,
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
            docs = [Document(content=text, meta={"name": str(filepath)})]

        elif document_creation_mode == "one-doc-per-page":
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            for el in elements:
                metadata = {"name": str(filepath)}
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                page_number = int(metadata.get("page_number", 1))

                texts_per_page[page_number] += str(el) + separator
                meta_per_page[page_number].update(metadata)

            docs = [Document(content=texts_per_page[page], meta=meta_per_page[page]) for page in texts_per_page.keys()]

        elif document_creation_mode == "one-doc-per-element":
            for el in elements:
                metadata = {"name": str(filepath)}
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                if hasattr(el, "category"):
                    metadata["category"] = el.category
                doc = Document(content=str(el), meta=metadata)
                docs.append(doc)

        return docs

    def _partition_file_into_elements(self, filepath: Path) -> List[Element]:
        """
        Partition a file into elements using the Unstructured API.
        """
        elements = []
        try:
            elements = partition_via_api(
                filename=str(filepath), api_url=self.api_url, api_key=self.api_key, **self.unstructured_kwargs
            )
        except Exception as e:
            logger.warning(f"Unstructured could not process file {filepath}. Error: {e}")
        return elements
