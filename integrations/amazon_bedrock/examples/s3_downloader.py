import os
from pathlib import Path

from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.downloader.s3 import S3Downloader
from haystack_integrations.common.amazon_bedrock.errors import AmazonS3Error

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
    "https://bucket_name.s3.eu-central-1.amazonaws.com/key_name/file_name.json" ## This will be skipped because it's not an s3:// URL
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

try:
    result = downloader.run(documents=docs, sources=sources)
except AmazonS3Error as e:
    print(f"Download failed: {e}")
else:
    print("Downloaded docs:")
    for d in result["documents"]:
        print(f"- {d.meta.get('file_name')} -> {d.meta.get('file_path')} (mime: {d.meta.get('mime_type')})")

    print("Sources:")
    for s in result["sources"]:
        print("-", s)
