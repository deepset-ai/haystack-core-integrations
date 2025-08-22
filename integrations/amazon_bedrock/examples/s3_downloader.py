import os
from pathlib import Path

from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.downloaders.s3 import S3Downloader

# Set up AWS credentials
# You can also set these as environment variables
aws_profile_name = os.environ.get("AWS_PROFILE") or "default"
aws_region_name = os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"

downloader = S3Downloader(
    aws_profile_name=Secret.from_token(aws_profile_name),
    aws_region_name=Secret.from_token(aws_region_name),
    download_dir=str(Path.cwd() / "s3_cache"),
    file_extensions=[".json"],  # only download .json files
    sources_target_type="str",  # return file paths as strings in result["sources"]
)

# Only s3:// URLs are accepted in this implementation
sources = [
    "s3://bucket_name/key_name/file_name.json",
    ## This will be skipped because it's not an s3:// URL
    "https://bucket_name.s3.eu-central-1.amazonaws.com/key_name/file_name.json",
]

docs = [
    # Using bucket + key
    Document(
        content="",
        meta={
            "s3_bucket": "bucket_name",
            "s3_key": "key_name/file_name.json",
            "file_name": "file_name.json",
        },
    ),
    # Using s3_url
    Document(
        content="",
        meta={
            "s3_url": "s3://bucket_name/key_name/file_name.json",
            "file_name": "opensearch-website.json",
        },
    ),
]


result = downloader.run(documents=docs, sources=sources)
