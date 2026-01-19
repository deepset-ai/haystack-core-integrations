# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.ai.documentintelligence.models import AnalyzeResult as DIAnalyzeResult
from azure.core.credentials import AzureKeyCredential
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class AzureDocumentIntelligenceConverter:
    """
    Converts files to Documents using Azure's Document Intelligence service.

    This component uses the azure-ai-documentintelligence package (v1.0.0+) and outputs
    GitHub Flavored Markdown for better integration with LLM/RAG applications.

    Supported file formats: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, HTML.

    Key features:
    - Markdown output with preserved structure (headings, tables, lists)
    - Inline table integration (tables rendered as markdown tables)
    - Improved layout analysis and reading order
    - Support for section headings

    To use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. For setup instructions, see
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

    ### Usage example

    ```python
    import os
    from haystack_integrations.components.converters.azure_doc_intelligence import (
        AzureDocumentIntelligenceConverter,
    )
    from haystack.utils import Secret

    converter = AzureDocumentIntelligenceConverter(
        endpoint=os.environ["AZURE_DI_ENDPOINT"],
        api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
    )

    results = converter.run(sources=["invoice.pdf", "contract.docx"])
    documents = results["documents"]

    # Documents contain markdown with inline tables
    print(documents[0].content)
    ```
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: Secret = Secret.from_env_var("AZURE_DI_API_KEY"),
        model_id: str = "prebuilt-document",
        store_full_path: bool = False,
    ):
        """
        Creates an AzureDocumentIntelligenceConverter component.

        :param endpoint:
            The endpoint URL of your Azure Document Intelligence resource.
            Example: "https://YOUR_RESOURCE.cognitiveservices.azure.com/"
        :param api_key:
            API key for Azure authentication. Can use Secret.from_env_var()
            to load from AZURE_DI_API_KEY environment variable.
        :param model_id:
            Azure model to use for analysis. Options:
            - "prebuilt-document": General document analysis (default)
            - "prebuilt-read": Fast OCR for text extraction
            - "prebuilt-layout": Enhanced layout analysis with better table/structure detection
            - Custom model IDs from your Azure resource
        :param store_full_path:
            If True, stores complete file path in metadata.
            If False, stores only the filename (default).
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_id = model_id
        self.store_full_path = store_full_path
        self.client: DocumentIntelligenceClient | None = None

    def warm_up(self):
        """
        Initializes the Azure Document Intelligence client.
        """
        if self.client is None:
            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key.resolve_value() or "")
            )

    @component.output_types(documents=list[Document], raw_azure_response=list[dict])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document] | list[dict]]:
        """
        Convert a list of files to Documents using Azure's Document Intelligence service.

        :param sources:
            List of file paths or ByteStream objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will be
            zipped. If `sources` contains ByteStream objects, their `meta` will be added to the output Documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents
            - `raw_azure_response`: List of raw Azure responses used to create the Documents
        """
        if self.client is None:
            self.warm_up()

        documents = []
        azure_responses = []
        meta_list: list[dict[str, Any]] = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            try:
                analyze_request = AnalyzeDocumentRequest(bytes_source=bytestream.data)

                poller = self.client.begin_analyze_document(  # type: ignore[union-attr]
                    model_id=self.model_id,
                    body=analyze_request,
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                result = poller.result()
                azure_responses.append(result.as_dict())

                merged_metadata = {**bytestream.meta, **metadata}
                if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(file_path)

                doc = self._create_document(result, merged_metadata)
                documents.append(doc)

            except Exception as e:
                logger.warning(
                    "Failed to convert {source} using Azure Document Intelligence. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

        return {"documents": documents, "raw_azure_response": azure_responses}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            endpoint=self.endpoint,
            model_id=self.model_id,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AzureDocumentIntelligenceConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _create_document(self, result: "DIAnalyzeResult", meta: dict[str, Any]) -> Document:
        """
        Create a Document from Azure Document Intelligence result.

        :param result: The DIAnalyzeResult from Azure Document Intelligence.
        :param meta: Metadata dictionary to attach to the document.
        :returns: A Document with markdown content.
        """
        markdown_content = result.content or ""

        doc_meta = {
            **meta,
            "model_id": self.model_id,
            "page_count": len(result.pages) if result.pages else 0,
        }

        return Document(content=markdown_content, meta=doc_meta)
