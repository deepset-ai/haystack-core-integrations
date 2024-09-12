# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses.byte_stream import ByteStream
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

    converter = UnstructuredFileConverter()
    documents = converter.run(paths = ["a/file/path.pdf", "a/directory/path"])["documents"]
    ```
    """

    def __init__(
        self,
        api_url: str = UNSTRUCTURED_HOSTED_API_URL,
        api_key: Optional[Secret] = Secret.from_env_var("UNSTRUCTURED_API_KEY", strict=False),  # noqa: B008
        document_creation_mode: Literal[
            "one-doc-per-file", "one-doc-per-page", "one-doc-per-element"
        ] = "one-doc-per-file",
        separator: str = "\n\n",
        unstructured_kwargs: Optional[Dict[str, Any]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "UnstructuredFileConverter":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        sources: Union[List[Union[str, os.PathLike, ByteStream]]], 
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        """
        Convert files or byte streams to Haystack Documents using the Unstructured API.

        :param sources: List of file paths or byte streams to convert.
            Paths can be files or directories. Byte streams are also supported.
        :param meta: Optional metadata to attach to the Documents.
            This value can be a single dictionary or a list of dictionaries, matching the number of sources.
        :returns: A dictionary with the following key:
            - `documents`: List of Haystack Documents.
        :raises ValueError: If `meta` is a list and `sources` contains directories.
        """

        # Separate file paths and byte streams
        filepaths, filepaths_in_directories, byte_streams = self._get_sources(sources)


        if filepaths_in_directories and isinstance(meta, list):
            error = """"If providing directories in the `paths` parameter,
             `meta` can only be a dictionary (metadata applied to every file),
             and not a list. To specify different metadata for each file,
             provide an explicit list of direct paths instead."""
            raise ValueError(error)

        # Combine file paths and directories for processing
        all_filepaths = filepaths + filepaths_in_directories
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(all_filepaths) + len(byte_streams))

        # Process file paths
        for filepath, metadata in tqdm(
            zip(all_filepaths, meta_list[:len(all_filepaths)]), desc="Converting files to Haystack Documents"
        ):
            elements = self._partition_source_into_elements(source=filepath)
            docs_for_file = self._create_documents(
                filepath=filepath,
                elements=elements,
                document_creation_mode=self.document_creation_mode,
                separator=self.separator,
                meta=metadata,
            )
            documents.extend(docs_for_file)

        # Process byte streams
        for bytestream in byte_streams:
            elements = self._partition_source_into_elements(source=bytestream)
            docs_for_stream = self._create_documents(
                elements=elements,
                document_creation_mode=self.document_creation_mode,
                separator=self.separator,
                meta=bytestream.meta,
            )
            documents.extend(docs_for_stream)

        return {"documents": documents}

    def _get_sources(
        self, 
        sources: Union[List[str], List[os.PathLike], List[ByteStream]]
    ) -> Tuple[List[Path], List[Path], List[ByteStream]]:
        """
        Helper function to process and return file paths, directories, and byte streams separately.
        """
        paths_obj = [Path(source) for source in sources if isinstance(source, (str, os.PathLike))]
        byte_streams = [source for source in sources if isinstance(source, ByteStream)]

        # Separate files and directories
        filepaths = [path for path in paths_obj if path.is_file()]
        directories = [path for path in paths_obj if path.is_dir()]

        filepaths_in_directories = [
            filepath for path in paths_obj if path.is_dir() for filepath in path.glob("*.*") if filepath.is_file()
        ]

        return filepaths, filepaths_in_directories, byte_streams

    @staticmethod
    def _create_documents(
        elements: List[Element],
        document_creation_mode: Literal["one-doc-per-file", "one-doc-per-page", "one-doc-per-element"],
        separator: str,
        meta: Dict[str, Any],
        filepath: Optional[Path] = None,
    ) -> List[Document]:
        """
        Create Haystack Documents from the elements returned by Unstructured.
        """
        docs = []

        if document_creation_mode == "one-doc-per-file":
            text = separator.join([str(el) for el in elements])
            metadata = copy.deepcopy(meta)
            if filepath:
                metadata["file_path"] = str(filepath)  # Only include file path if provided
            docs = [Document(content=text, meta=metadata)]

        elif document_creation_mode == "one-doc-per-page":
            texts_per_page: defaultdict[int, str] = defaultdict(str)
            meta_per_page: defaultdict[int, dict] = defaultdict(dict)
            for el in elements:
                metadata = copy.deepcopy(meta)
                if filepath:
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
                if filepath:
                    metadata["file_path"] = str(filepath)
                metadata["element_index"] = index
                if hasattr(el, "metadata"):
                    metadata.update(el.metadata.to_dict())
                if hasattr(el, "category"):
                    metadata["category"] = el.category
                doc = Document(content=str(el), meta=metadata)
                docs.append(doc)

        return docs


    def _partition_source_into_elements(self, source: Union[Path, ByteStream]) -> List[Element]:
        """
        Partition a file into elements using the Unstructured API.
        """
        elements = []
        try:
            if isinstance(source, Path):
                elements = partition_via_api(
                    filename=str(source),
                    api_url=self.api_url,
                    api_key=self.api_key.resolve_value() if self.api_key else None,
                    **self.unstructured_kwargs,
                )
            else:
                elements = partition_via_api(
                    file=source.data,
                    metadata_filename=str(source.meta),
                    api_url=self.api_url,
                    api_key=self.api_key.resolve_value() if self.api_key else None,
                    **self.unstructured_kwargs,
                )
        except Exception as e:
            logger.warning(f"Unstructured could not process source {source}. Error: {e}")
        return elements
