# amazon-textract-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/amazon-textract-haystack.svg)](https://pypi.org/project/amazon-textract-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/amazon-textract-haystack.svg)](https://pypi.org/project/amazon-textract-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/amazon_textract/CHANGELOG.md)

---

## Overview

A [Haystack](https://haystack.deepset.ai/) integration for [AWS Textract](https://aws.amazon.com/textract/) that extracts text and structured data from documents using OCR.

The `AmazonTextractConverter` component converts images and single-page PDFs into Haystack `Document` objects using the AWS Textract synchronous API.

**Supported file formats:** JPEG, PNG, TIFF, BMP, and single-page PDF (up to 10 MB).

## Installation

```bash
pip install amazon-textract-haystack
```

## Usage

### Basic text extraction

Extract plain text from a document using `DetectDocumentText`:

```python
from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

converter = AmazonTextractConverter()
results = converter.run(sources=["document.png"])
documents = results["documents"]

print(documents[0].content)
```

### Table and form analysis

Use `AnalyzeDocument` to detect tables and forms by setting `feature_types`:

```python
converter = AmazonTextractConverter(feature_types=["TABLES", "FORMS"])
results = converter.run(sources=["invoice.png"])

documents = results["documents"]
raw_responses = results["raw_textract_response"]
```

Valid `feature_types` values: `"TABLES"`, `"FORMS"`, `"SIGNATURES"`, `"LAYOUT"`.

### Natural-language queries

Ask questions about a document and get extracted answers. The `QUERIES` feature type
is enabled automatically when you pass the `queries` parameter at runtime:

```python
converter = AmazonTextractConverter()
results = converter.run(
    sources=["medical_form.png"],
    queries=["What is the patient name?", "What is the date of birth?"],
)

documents = results["documents"]
raw_responses = results["raw_textract_response"]
```

Queries can be combined with `feature_types` for both structural and question-based extraction:

```python
converter = AmazonTextractConverter(feature_types=["TABLES", "FORMS"])
results = converter.run(
    sources=["invoice.png"],
    queries=["What is the total amount due?"],
)
```

### In a Haystack pipeline

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner
from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

pipeline = Pipeline()
pipeline.add_component("converter", AmazonTextractConverter())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.connect("converter.documents", "cleaner.documents")

result = pipeline.run({"converter": {"sources": ["scan.png"]}})
```

## AWS Credentials

The component uses the standard boto3 credential chain. You can configure credentials in any of these ways:

1. **Environment variables** (default): Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION`.
2. **AWS credentials file**: Configure via `~/.aws/credentials` and `~/.aws/config`.
3. **IAM role**: When running on AWS infrastructure (EC2, Lambda, ECS).
4. **Explicit parameters**:

```python
from haystack.utils import Secret

converter = AmazonTextractConverter(
    aws_access_key_id=Secret.from_env_var("MY_AWS_KEY"),
    aws_secret_access_key=Secret.from_env_var("MY_AWS_SECRET"),
    aws_region_name=Secret.from_token("us-east-1"),
)
```

## Running Tests

Unit tests (no AWS credentials needed):

```bash
cd integrations/amazon_textract
hatch run test:unit
```

Integration tests (require AWS credentials and a test image at `tests/test_files/sample_text.png`):

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
hatch run test:integration
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
