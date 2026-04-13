# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.converters.amazon_textract.errors import (
    AmazonTextractConfigurationError,
)

logger = logging.getLogger(__name__)

VALID_FEATURE_TYPES = frozenset({"TABLES", "FORMS", "SIGNATURES", "LAYOUT"})


@component
class AmazonTextractConverter:
    """
    Converts documents to Haystack Documents using AWS Textract.

    This component uses AWS Textract to extract text and optionally structured data
    (tables, forms) from images and single-page PDFs.

    When `feature_types` is not set, the component uses `DetectDocumentText` for
    plain text OCR. When `feature_types` is set (e.g. `["TABLES", "FORMS"]`), it
    uses `AnalyzeDocument` for richer structural analysis.

    Natural-language queries are also supported via the `queries` parameter on
    `run()`. When queries are provided, the `QUERIES` feature type is added
    automatically and Textract returns answers extracted from the document.

    Supported input formats: JPEG, PNG, TIFF, BMP, and single-page PDF (up to 10 MB).

    AWS credentials are resolved via `Secret` parameters or the default boto3
    credential chain (environment variables, AWS config files, IAM roles).

    ### Usage example

    ```python
    from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

    converter = AmazonTextractConverter()
    results = converter.run(sources=["document.png"])
    documents = results["documents"]
    ```
    """

    def __init__(
        self,
        *,
        aws_access_key_id: Secret | None = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),
        aws_secret_access_key: Secret | None = Secret.from_env_var("AWS_SECRET_ACCESS_KEY", strict=False),
        aws_session_token: Secret | None = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),
        aws_region_name: Secret | None = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),
        aws_profile_name: Secret | None = Secret.from_env_var("AWS_PROFILE", strict=False),
        feature_types: list[str] | None = None,
        store_full_path: bool = False,
        boto3_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an AmazonTextractConverter component.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name. Must be a region that supports Textract.
        :param aws_profile_name: AWS profile name from the credentials file.
        :param feature_types:
            List of feature types to detect when using AnalyzeDocument.
            Valid values: "TABLES", "FORMS", "SIGNATURES", "LAYOUT".
            If None, uses DetectDocumentText for basic text extraction.
            The "QUERIES" feature type is managed automatically when the
            `queries` parameter is passed to `run()`.
        :param store_full_path:
            If True, stores the complete file path in Document metadata.
            If False, stores only the filename (default).
        :param boto3_config:
            Dictionary of configuration options for the underlying boto3 client.
            Can be used to tune retry behavior, timeouts, and connection management.
        """
        if feature_types is not None:
            invalid = set(feature_types) - VALID_FEATURE_TYPES
            if invalid:
                msg = f"Invalid feature_types: {invalid}. Valid values are: {sorted(VALID_FEATURE_TYPES)}"
                raise ValueError(msg)

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.feature_types = feature_types
        self.store_full_path = store_full_path
        self.boto3_config = boto3_config
        self._client: Any = None

    def warm_up(self) -> None:
        """Initializes the AWS Textract client."""
        if self._client is not None:
            return

        def resolve_secret(secret: Secret | None) -> str | None:
            return secret.resolve_value() if secret else None

        try:
            session = boto3.Session(
                aws_access_key_id=resolve_secret(self.aws_access_key_id),
                aws_secret_access_key=resolve_secret(self.aws_secret_access_key),
                aws_session_token=resolve_secret(self.aws_session_token),
                region_name=resolve_secret(self.aws_region_name),
                profile_name=resolve_secret(self.aws_profile_name),
            )
            config = Config(
                user_agent_extra="x-client-framework:haystack",
                **(self.boto3_config if self.boto3_config else {}),
            )
            self._client = session.client("textract", config=config)
        except Exception as e:
            msg = (
                "Could not connect to AWS Textract. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonTextractConfigurationError(msg) from e

    @component.output_types(documents=list[Document], raw_textract_response=list[dict])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
        queries: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Convert documents to Haystack Documents using AWS Textract.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources.
        :param queries:
            Optional list of natural-language questions to ask about each document.
            When provided, the Textract ``QUERIES`` feature type is enabled
            automatically and each question is sent as a query. Answers are
            included in the raw Textract response. Example:
            ``["What is the patient name?", "What is the total due?"]``
        :returns:
            A dictionary with the following keys:
            - `documents`: List of created Documents with extracted text as content.
            - `raw_textract_response`: List of raw Textract API responses.
        """
        if self._client is None:
            self.warm_up()

        documents: list[Document] = []
        raw_responses: list[dict[str, Any]] = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source=source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            try:
                response = self._call_textract(bytestream.data, queries=queries)
                raw_responses.append(response)

                merged_metadata = {**bytestream.meta, **metadata}
                if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                    merged_metadata["file_path"] = os.path.basename(file_path)

                doc = self._create_document(response, merged_metadata)
                documents.append(doc)

            except (BotoCoreError, ClientError) as e:
                logger.warning(
                    "Failed to convert {source} using AWS Textract. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

        return {"documents": documents, "raw_textract_response": raw_responses}

    def _call_textract(self, document_bytes: bytes, queries: list[str] | None = None) -> dict[str, Any]:
        """Calls the appropriate Textract API based on configuration."""
        doc_param: dict[str, Any] = {"Document": {"Bytes": document_bytes}}

        feature_types = list(self.feature_types) if self.feature_types else []
        if queries:
            if "QUERIES" not in feature_types:
                feature_types.append("QUERIES")

        if feature_types:
            kwargs: dict[str, Any] = {**doc_param, "FeatureTypes": feature_types}
            if queries:
                kwargs["QueriesConfig"] = {"Queries": [{"Text": q} for q in queries]}
            return self._client.analyze_document(**kwargs)

        return self._client.detect_document_text(**doc_param)

    def _create_document(self, response: dict[str, Any], meta: dict[str, Any]) -> Document:
        """
        Creates a Document from a Textract response.

        Extracts LINE blocks in reading order and joins them with newlines.
        """
        blocks = response.get("Blocks", [])
        lines = [block["Text"] for block in blocks if block.get("BlockType") == "LINE" and "Text" in block]
        content = "\n".join(lines)

        page_count = sum(1 for block in blocks if block.get("BlockType") == "PAGE")

        doc_meta = {
            **meta,
            "page_count": page_count,
        }

        return Document(content=content, meta=doc_meta)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            feature_types=self.feature_types,
            store_full_path=self.store_full_path,
            boto3_config=self.boto3_config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmazonTextractConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            keys=[
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
                "aws_profile_name",
            ],
        )
        return default_from_dict(cls, data)
