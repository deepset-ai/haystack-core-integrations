# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from azure.ai.documentintelligence.models import AnalyzeResult as DIAnalyzeResult
from azure.core.credentials import AzureKeyCredential
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from pandas import DataFrame

logger = logging.getLogger(__name__)


@component
class AzureDocumentIntelligenceConverter:
    """
    Converts files to Documents using Azure's Document Intelligence service (2024 API).

    This component uses the latest azure-ai-documentintelligence package and supports
    markdown output for better integration with LLM/RAG applications.

    Supported file formats: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, HTML.

    Key features:
    - Markdown output with preserved structure (headings, tables, lists)
    - Inline table integration (no separate table documents)
    - Improved layout analysis and reading order
    - Better table extraction
    - Support for section headings

    To use this component, you need an active Azure account
    and a Document Intelligence or Cognitive Services resource. For help with setting up your resource, see
    [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api).

    ### Usage example

    ```python
    import os
    from haystack_integrations.components.converters.azure_doc_intelligence import (
        AzureDocumentIntelligenceConverter,
    )
    from haystack.utils import Secret

    # Basic usage with markdown output (recommended for RAG)
    converter = AzureDocumentIntelligenceConverter(
        endpoint=os.environ["AZURE_DI_ENDPOINT"],
        api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
        output_format="markdown"
    )

    results = converter.run(sources=["invoice.pdf", "contract.docx"])
    documents = results["documents"]

    # Documents contain markdown with inline tables
    print(documents[0].content)
    # Output:
    # # Invoice
    #
    # | Item | Quantity | Price |
    # |------|----------|-------|
    # | Widget | 10 | $50.00 |
    #
    # Total: $500.00

    # For backward compatibility, use text mode with CSV tables
    converter_compat = AzureDocumentIntelligenceConverter(
        endpoint=os.environ["AZURE_DI_ENDPOINT"],
        api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
        output_format="text",
        table_format="csv"
    )
    ```
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: Secret = Secret.from_env_var("AZURE_AI_API_KEY"),
        model_id: str = "prebuilt-read",
        output_format: Literal["text", "markdown"] = "markdown",
        table_format: Literal["csv", "markdown"] = "markdown",
        store_full_path: bool = False,
    ):
        """
        Creates an AzureDocumentIntelligenceConverter component.

        :param endpoint:
            The endpoint URL of your Azure Document Intelligence resource.
            Example: "https://YOUR_RESOURCE.cognitiveservices.azure.com/"
        :param api_key:
            API key for Azure authentication. Can use Secret.from_env_var()
            to load from AZURE_AI_API_KEY environment variable.
        :param model_id:
            Azure model to use for analysis. Options:
            - "prebuilt-read": Fast OCR for text extraction (default)
            - "prebuilt-layout": Enhanced layout analysis with better table/structure detection
            - "prebuilt-document": General document analysis
            - Custom model IDs from your Azure resource
        :param output_format:
            Output format for document content.
            - "markdown": Returns GitHub Flavored Markdown with inline tables (recommended for RAG)
            - "text": Returns plain text with optional separate table documents
        :param table_format:
            How to format tables when output_format="text".
            - "markdown": Format tables as markdown (inline)
            - "csv": Format tables as CSV in separate documents
            Ignored when output_format="markdown" (tables are already in markdown).
        :param store_full_path:
            If True, stores complete file path in metadata.
            If False, stores only the filename (default).
        """
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(api_key.resolve_value() or "")
        )
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_id = model_id
        self.output_format = output_format
        self.table_format = table_format
        self.store_full_path = store_full_path

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
                # Determine output format
                content_format = DocumentContentFormat.MARKDOWN if self.output_format == "markdown" else None

                # Create analyze request
                analyze_request = AnalyzeDocumentRequest(bytes_source=bytestream.data)

                # Call Azure API
                poller = self.client.begin_analyze_document(
                    model_id=self.model_id, body=analyze_request, output_content_format=content_format
                )
                result = poller.result()
                azure_responses.append(result.as_dict())

                # Merge metadata
                merged_metadata = {**bytestream.meta, **metadata}
                if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(file_path)

                # Process based on output format
                if self.output_format == "markdown":
                    doc = self._process_markdown_result(result, merged_metadata)
                    documents.append(doc)
                else:
                    docs = self._process_text_result(result, merged_metadata)
                    documents.extend(docs)

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
            output_format=self.output_format,
            table_format=self.table_format,
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

    def _process_markdown_result(self, result: "DIAnalyzeResult", meta: dict[str, Any]) -> Document:
        """
        Process result when output_format='markdown'.

        :param result: The DIAnalyzeResult from Azure Document Intelligence.
        :param meta: Metadata dictionary to attach to the document.
        :returns: A single Document with markdown content.
        """
        # Azure returns complete markdown in result.content
        markdown_content = result.content or ""

        # Build metadata
        doc_meta = {
            **meta,
            "content_format": "markdown",
            "model_id": self.model_id,
            "page_count": len(result.pages) if result.pages else 0,
        }

        return Document(content=markdown_content, meta=doc_meta)

    def _process_text_result(self, result: "DIAnalyzeResult", meta: dict[str, Any]) -> list[Document]:
        """
        Process result when output_format='text'.

        :param result: The DIAnalyzeResult from Azure Document Intelligence.
        :param meta: Metadata dictionary to attach to the documents.
        :returns: List of Documents (text + optional table documents).
        """
        documents = []

        # Extract tables if table_format='csv'
        if self.table_format == "csv" and result.tables:
            table_docs = self._extract_csv_tables(result, meta)
            documents.extend(table_docs)

        # Extract main text content
        text_doc = self._extract_text_content(result, meta)
        documents.append(text_doc)

        return documents

    def _extract_text_content(self, result: "DIAnalyzeResult", meta: dict[str, Any]) -> Document:
        """
        Extract text from paragraphs.

        :param result: The DIAnalyzeResult from Azure Document Intelligence.
        :param meta: Metadata dictionary to attach to the document.
        :returns: A single Document with all text.
        """
        # Group paragraphs by page
        pages_text = []
        if result.paragraphs:
            page_to_paragraphs: dict[int, list[str]] = defaultdict(list)
            for para in result.paragraphs:
                page_num = para.bounding_regions[0].page_number if para.bounding_regions else 1
                # Skip paragraphs that are part of tables if extracting tables separately
                if self.table_format == "csv" and self._is_paragraph_in_table(para, result.tables):
                    continue
                page_to_paragraphs[page_num].append(para.content)

            # Join paragraphs with page breaks
            max_page = max(page_to_paragraphs.keys()) if page_to_paragraphs else 0
            for page_num in range(1, max_page + 1):
                page_text = "\n".join(page_to_paragraphs.get(page_num, []))
                pages_text.append(page_text)

        all_text = "\f".join(pages_text)
        return Document(content=all_text, meta={**meta, "content_format": "text"})

    def _extract_csv_tables(self, result: "DIAnalyzeResult", meta: dict[str, Any]) -> list[Document]:
        """
        Extract tables as CSV (backward compatibility mode).

        :param result: The DIAnalyzeResult from Azure Document Intelligence.
        :param meta: Metadata dictionary to attach to the documents.
        :returns: List of Documents containing table CSV content.
        """
        table_documents: list[Document] = []

        if not result.tables:
            return table_documents

        for table in result.tables:
            # Build table as 2D array
            table_data = [[""] * table.column_count for _ in range(table.row_count)]

            for cell in table.cells:
                # Remove selection markers
                content = cell.content.replace(":selected:", "").replace(":unselected:", "")

                # Handle cell spanning
                column_span = cell.column_span if cell.column_span else 1
                row_span = cell.row_span if cell.row_span else 1

                for r in range(row_span):
                    for c in range(column_span):
                        row_idx = cell.row_index + r
                        col_idx = cell.column_index + c
                        if row_idx < table.row_count and col_idx < table.column_count:
                            table_data[row_idx][col_idx] = content

            # Convert to CSV
            df = DataFrame(data=table_data)
            csv_content = df.to_csv(header=False, index=False, lineterminator="\n")

            # Build metadata
            table_meta = {**meta, "table_format": "csv", "content_format": "table"}

            if table.bounding_regions:
                table_meta["page"] = table.bounding_regions[0].page_number

            table_documents.append(Document(content=csv_content, meta=table_meta))

        return table_documents

    def _is_paragraph_in_table(self, paragraph: Any, tables: list | None) -> bool:
        """
        Check if a paragraph is part of a table.

        :param paragraph: Paragraph object to check.
        :param tables: List of table objects.
        :returns: True if paragraph is in a table, False otherwise.
        """
        if not tables or not paragraph.spans:
            return False

        para_offset = paragraph.spans[0].offset
        para_length = paragraph.spans[0].length
        para_end = para_offset + para_length

        for table in tables:
            if not table.spans:
                continue
            table_offset = table.spans[0].offset
            table_end = table_offset + table.spans[0].length

            # Check if paragraph overlaps with table
            if table_offset <= para_offset <= table_end or table_offset <= para_end <= table_end:
                return True

        return False
