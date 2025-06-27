# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from textractor import Textractor

logger = logging.getLogger(__name__)


@component
class AmazonTextractDocumentConverter:
    """
    A component for converting files to Haystack Documents using Amazon Textract DetectDocumentText API.

    This component supports the extraction of text from PNG, JPEG, TIFF, and PDF files.
    Supported sources include both local file paths and S3 URLs. For multipage documents, input must be provided
    via S3 URI, as Amazon Textract's asynchronous APIs are required for multipage documents.
    See [documentation](https://docs.aws.amazon.com/textract/latest/dg/API_AnalyzeDocument.html#API_AnalyzeDocument_RequestSyntax)
    for more information.

    ### Usage example

    ```python
    from haystack_integrations.components.generators.amazon_textract import AmazonTextractDocumentConverter

    converter = AmazonTextractDocumentConverter()

    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    ```

    AmazonTextractDocumentConverter uses AWS for authentication. You can use the AWS CLI to authenticate through
    your IAM. For more information on setting up an IAM identity-based policy, see [Amazon Textract documentation]
    (https://docs.aws.amazon.com/textract/latest/dg/security_iam_id-based-policy-examples.html).
    If the AWS environment is configured correctly, the AWS credentials are not required as they're loaded
    automatically from the environment or the AWS configuration file.
    If the AWS environment is not configured, set `aws_access_key_id`, `aws_secret_access_key`,
    `aws_session_token`, and `aws_region_name` as environment variables. Make sure the region you set
    supports Amazon Textract.
    """

    def __init__(
        self,
        aws_region_name: Optional[Secret] = Secret.from_env_var("AWS_REGION", strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
    ):
        """
        Create a AmazonTextractDocumentConverter instance.

        :param aws_region_name: The AWS region name. Make sure the region you set supports Amazon Textract.
        :param aws_profile_name: The AWS profile name.
        """
        self._aws_profile_name = aws_profile_name.resolve_value() if aws_profile_name else None
        self._aws_region_name = aws_region_name.resolve_value() if aws_region_name else None
        self._extractor = Textractor(profile_name=self._aws_profile_name, region_name=self._aws_region_name)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
    ) -> dict[str, list[Document]]:
        """
        Converts a list of file paths or S3 URLs to Haystack Documents.

        :param sources: The list of source files or S3 URIs.
        :param meta: Optional metadata to attach to the Documents.
        :returns: A dictionary with the following key:
            - `documents`: List of Haystack Documents.
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        # TODO: Handle multipage documents
        # TODO: Handle s3 files
        # TODO: Handle PIL.Image
        for source, metadata in zip(sources, meta_list):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                msg = "Could not read {source}. Skipping it. Error: {error}"
                logger.warning(msg, source=source, error=e)
                continue

            textract_document = self._extractor.detect_document_text(file_source=source)

            content = textract_document.text
            # TODO: Handle metadata merging from textract document
            merged_metadata = {**bytestream.meta, **metadata}
            doc = Document(content=content, meta=merged_metadata)

            documents.append(doc)

        return {"documents": documents}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AmazonTextractDocumentConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            aws_region_name=self._aws_region_name.to_dict() if self._aws_region_name else None,
            aws_profile_name=self._aws_profile_name.to_dict() if self._aws_profile_name else None,
        )
