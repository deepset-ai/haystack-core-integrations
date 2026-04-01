# presidio-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/presidio-haystack.svg)](https://pypi.org/project/presidio-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/presidio-haystack.svg)](https://pypi.org/project/presidio-haystack)

Haystack integration for [Microsoft Presidio](https://microsoft.github.io/presidio/) — PII detection and anonymization.

---

## Installation

```bash
pip install presidio-haystack
```

You also need to download the spaCy model used by Presidio:

```bash
python -m spacy download en_core_web_lg
```

## Components

- **PresidioDocumentCleaner** — anonymizes PII in `list[Document]`
- **PresidioTextCleaner** — anonymizes PII in `list[str]` (useful for query sanitization)
- **PresidioEntityExtractor** — detects PII entities and stores them in Document metadata

## Usage

```python
from haystack import Document
from haystack_integrations.components.preprocessors.presidio import PresidioDocumentCleaner

cleaner = PresidioDocumentCleaner()
result = cleaner.run(documents=[Document(content="My name is John, email: john@example.com")])
print(result["documents"][0].content)
# My name is <PERSON>, email: <EMAIL_ADDRESS>
```

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
