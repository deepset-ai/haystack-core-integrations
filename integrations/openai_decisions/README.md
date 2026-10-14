# openai-decisions-haystack

Haystack integration for the [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions).

The integration provides two components:

- `OpenAIDecisionsDocumentClassifier` for classifying documents using predicate, choice, and score questions.
- `OpenAIDecisionsTextRouter` for routing text to a predefined category.

## Installation

```bash
pip install openai-decisions-haystack
```

Set the `OPENAI_API_KEY` environment variable before using either component. An API key can also be provided explicitly through a Haystack `Secret`.

The default model is `gpt-6-luna`.

## Document classification

`OpenAIDecisionsDocumentClassifier` evaluates one or more questions for each document and stores the answers in the document metadata.

```python
from haystack import Document
from haystack_integrations.components.classifiers.openai_decisions import (
    OpenAIDecisionsDocumentClassifier,
)

classifier = OpenAIDecisionsDocumentClassifier(
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "choices": [
                {
                    "value": "billing",
                    "description": "Payments, invoices, and refunds",
                },
                {
                    "value": "technical",
                    "description": "Bugs and product errors",
                },
            ],
        },
        "refund": {
            "type": "predicate",
            "instructions": "Is the customer asking for a refund?",
        },
    }
)

result = classifier.run(documents=[Document(content="I was charged twice. Please refund me.")])

print(result["documents"][0].meta["openai_decisions"])
```

Each question must have a unique name, a supported type, and instructions. Choice questions require `choices`, while score questions require `levels`.

The component returns `documents` and `failed_documents`. Successful answers are stored under `meta["openai_decisions"]` by default. The metadata key can be changed using `metadata_field`.

By default, documents that fail classification are returned in `failed_documents`. Set `raise_on_failure=True` to propagate errors instead.

## Text routing

`OpenAIDecisionsTextRouter` selects a label for an input string. Each label becomes a Haystack output socket.

```python
from haystack_integrations.components.routers.openai_decisions import (
    OpenAIDecisionsTextRouter,
)

router = OpenAIDecisionsTextRouter(
    labels={
        "billing": "Payments, invoices, and refunds",
        "technical": "Bugs and product errors",
    },
    instructions="Which team should handle this message?",
    min_confidence=0.6,
)

result = router.run(text="I was charged twice.")
print(result)
```

When `min_confidence` is configured, the router adds a `low_confidence` output socket. Predictions below the configured confidence threshold are sent to that output.

## Image inputs

The document classifier supports Haystack `ImageContent` objects included in a document's content or in the field specified by `classification_field`.

Images containing base64 data are converted to inline data URLs. Images provided through local file paths are read and encoded before being sent to the API. URL-backed `ImageContent` objects are not supported by this implementation.

The integration does not upload files or use OpenAI file IDs.

For supported input formats and question types, see the [OpenAI Decisions API guide](https://developers.openai.com/api/docs/guides/decisions).