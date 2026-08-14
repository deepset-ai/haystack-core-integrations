# amazon-bedrock-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/amazon-bedrock-haystack.svg)](https://pypi.org/project/amazon-bedrock-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/amazon-bedrock-haystack.svg)](https://pypi.org/project/amazon-bedrock-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/amazon-bedrock)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/amazon_bedrock/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to authenticate with AWS.
For example, you can do that by exporting the following environment variables:

```console
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...  # only for temporary credentials
export AWS_DEFAULT_REGION=...
```

Some tests target specific AWS resources and are skipped unless the corresponding
environment variables are set:

- `AWS_BEDROCK_GUARDRAIL_ID` and `AWS_BEDROCK_GUARDRAIL_VERSION` — guardrail tests.
- `AWS_KNOWLEDGE_BASE_ID` — Knowledge Base retriever tests.
- `S3_DOWNLOADER_BUCKET` — S3 downloader tests.
- `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION` — Bedrock inference generator tests.
