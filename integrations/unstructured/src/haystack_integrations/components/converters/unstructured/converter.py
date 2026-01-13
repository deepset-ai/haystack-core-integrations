# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.utils import Secret, deserialize_secrets_inplace
from tqdm import tqdm

from unstructured.documents.elements import Element
from unstructured.partition.api import partition_via_api

logger = logging.getLogger(__name__)

UNSTRUCTURED_HOSTED_API_URL = "https://api.unstructured.io/general/v0/general"


@component
class UnstructuredFileConverter:
    """
    A component for converting files to Haystack Documents using the Unstructured API (hosted or running locally).

    For the supported file types and the specific API parameters, see
    [Unstructured docs](https://docs.unstructured.io/api-reference/api-services/overview).

    Usage example:
    ```python
    from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter

    # make sure to either set the environment variable UNSTRUCTURED_API_KEY
    # or run the Unstructured API locally:
    # docker run -p 8000:8000 -d --rm --name unstructured-api quay.io/unstructured-io/unstructured-api:latest
    # --port 8000 --host 0.0.0.0

    converter = UnstructuredFileConverter(
        # api_url="http://localhost:8000/general/v0/general"  # <-- Uncomment this if running Unstructured locally
    )
    documents = converter.run(paths = ["a/file/path.pdf", "a/directory/path"])["documents"]
    ```
    """

    def __init__(
        self,
        api_url: str = UNSTRUCTURED_HOSTED_API_URL,
        api_key: Secret | None = Secret.from_env_var("UNSTRUCTURED_API_KEY", strict=False),  # noqa: B008
        document_creation_mode: Literal[
            "one-doc-per-file", "one-doc-per-page", "one-doc-per-element"
        ] = "one-doc-per-file",
        separator: str = "\n\n",
        unstructured_kwargs: dict[str, Any] | None = None,
        progress_bar: bool = True,  # noqa: FBT001, FBT002
    ):
        """
        :param api_url: URL of the Unstructured API. Defaults to the URL of the hosted version.
            If you run the API locally, specify the URL of your local API (e.g. `"http://localhost:8000/general/v0/general"`).
        :param api_key: API key for the Unstructured API.
            It can be explicitly passed or read the environment variable `UNSTRUCTURED_API_KEY` (recommended).
            If you run the API locally, it is not needed.
        :param document_creation_mode: How to create Haystack Documents from the elements returned by Unstructured.
        `"one-doc-per-file"`: One Haystack Document per file. All elements are concatenated into one text field.
        `"one-doc-per-page"`: One Haystack Document per page.
        All elements on a page are concatenated into one text field.
        `"one-doc-per-element"`: One Haystack Document per element. Each element is converted to a Haystack Document.
        :param separator: Separator between elements when concatenating them into one text field.
        :param unstructured_kwargs: Additional parameters that are passed to the Unstructured API.
            For the available parameters, see
            [Unstructured API docs](https://docs.unstructured.io/api-reference/api-services/api-parameters).
        :param progress_bar: Whether to show a progress bar during the conversion.
        """

        self.api_url = api_url
        self.api_key = api_key
        self.document_creation_mode = document_creation_mode
        self.unstructured_kwargs = unstructured_kwargs or {}
        self.separator = separator
        self.progress_bar = progress_bar

        is_hosted_api = api_url == UNSTRUCTURED_HOSTED_API_URL

        # we check whether api_key is None or an empty string
        api_key_value = api_key.resolve_value() if api_key else None
        if is_hosted_api and not api_key_value:
            msg = (
                "To use the hosted version of Unstructured, you need to set the environment variable "
                "UNSTRUCTURED_API_KEY (recommended) or explicitly pass the parameter api_key."
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        return default_to_dict(
            self,
            api_url=self.api_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            document_creation_mode=self.document_creation_mode,
            separator=self.separator,
            unstructured_kwargs=self.unstructured_kwargs,
            progress_bar=self.progress_bar,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnstructuredFileConverter":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        paths: list[str] | list[os.PathLike],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert files to Haystack Documents using the Unstructured API.

        :param paths: List of paths to convert. Paths can be files or directories.
            If a path is a directory, all files in the directory are converted. Subdirectories are ignored.
        :param meta: Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of paths, because the two lists will be zipped.
            Please note that if the paths contain directories, `meta` can only be a single dictionary
            (same metadata for all files).

        :returns: A dictionary with the following key:
            - `documents`: List of Haystack Documents.

        :raises ValueError: If `meta` is a list and `paths` contains directories.
        """
        paths_obj = [Path(path) for path in paths]
        filepaths = [path for path in paths_obj if path.is_file()]
        filepaths_in_directories = [
            filepath for path in paths_obj if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()
        ]
        if filepaths_in_directories and isinstance(meta, list):
            error = """"If providing directories in the `paths` parameter,
             `meta` can only be a dictionary (metadata applied to every file),
             and not a list. To specify different metadata for each file,
             provide an explicit list of direct paths instead."""
            raise ValueError(error)

        all_filepaths = filepaths + filepaths_in_directories
        # currently, the files are converted sequentially to gently handle API failures
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(all_filepaths))

        for filepath, metadata in tqdm(
            zip(all_filepaths, meta_list, strict=True),
            desc="Converting files to Haystack Documents",
            disable=not self.progress_bar,
        ):
            elements = self._partition_file_into_elements(filepath=filepath)
            docs_for_file = self._create_documents(
                filepath=filepath,
                elements=elements,
                document_creation_mode=self.document_creation_mode,
                separator=self.separator,
                meta=metadata,
            )
            documents.extend(docs_for_file)
        return {"documents": documents}

    @staticmethod
    def _create_documents(
        filepath: Path,
        elements: list[Element],
        document_creation_mode: Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"],
        separator: str,
        meta: dict[str, Any],
    ) -> list[Document]:
        """
        Create Haystack Documents from the elements returned by Unstructured.
        """
        docs = []

        if document_creation_mode == "one-doc-per-file":
            text = separator.join([str(el) for el in elements])
            metadata = copy.deepcopy(meta)
            metadata["file_path"] = str(filepath)
            docs = [Document(content=text, meta=metadata)]

        elif document_creation_mode == "one-doc-per-page":
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            for el in elements:
                metadata = copy.deepcopy(meta)
                metadata["file_path"] = str(filepath)
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                page_number = int(metadata.get("page_number", 1))

                texts_per_page[page_number] += str(el) + separator
                meta_per_page[page_number].update(metadata)

            docs = [Document(content=texts_per_page[page], meta=meta_per_page[page]) for page in texts_per_page.keys()]

        elif document_creation_mode == "one-doc-per-element":
            for index, el in enumerate(elements):
                metadata = copy.deepcopy(meta)
                metadata["file_path"] = str(filepath)
                metadata["element_index"] = index
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                if hasattr(el, "category"):
                    metadata["category"] = el.category
                doc = Document(content=str(el), meta=metadata)
                docs.append(doc)
        return docs

    def _partition_file_into_elements(self, filepath: Path) -> list[Element]:
        """
        Partition a file into elements using the Unstructured API.
        """
        elements = []

        resolved_api_key = ""
        if self.api_key:
            resolved_api_key = self.api_key.resolve_value() or ""

        try:
            elements = partition_via_api(
                filename=str(filepath),
                api_url=self.api_url,
                api_key=resolved_api_key,
                **self.unstructured_kwargs,
            )
        except Exception as e:
            logger.warning(f"Unstructured could not process file {filepath}. Error: {e}")
        return elements
