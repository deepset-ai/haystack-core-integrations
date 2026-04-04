# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)


@component
class PresidioDocumentCleaner:
    """
    Anonymizes PII in Haystack Documents using [Microsoft Presidio](https://microsoft.github.io/presidio/).

    Accepts a list of Documents, detects personally identifiable information (PII) in their
    text content, and returns new Documents with PII replaced by entity type placeholders
    (e.g. `<PERSON>`, `<EMAIL_ADDRESS>`). Original Documents are not mutated.

    Documents without text content are passed through unchanged.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.preprocessors.presidio import PresidioDocumentCleaner

    cleaner = PresidioDocumentCleaner()
    result = cleaner.run(documents=[Document(content="My name is John and my email is john@example.com")])
    print(result["documents"][0].content)
    # My name is <PERSON> and my email is <EMAIL_ADDRESS>
    ```
    """

    def __init__(
        self,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
    ) -> None:
        """
        Initializes the PresidioDocumentCleaner.

        :param language:
            Language code for PII detection. Defaults to `"en"`.
        :param entities:
            List of PII entity types to detect and anonymize (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
            If `None`, all supported entity types are used.
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Anonymizes PII in the provided Documents.

        :param documents:
            List of Documents whose text content will be anonymized.
        :returns:
            A dictionary with key `documents` containing the cleaned Documents.
        """
        cleaned: list[Document] = []
        for doc in documents:
            if doc.content is None:
                cleaned.append(doc)
                continue
            try:
                analyzer_results = self._analyzer.analyze(
                    text=doc.content,
                    language=self.language,
                    entities=self.entities,
                    score_threshold=self.score_threshold,
                )
                anonymized = self._anonymizer.anonymize(text=doc.content, analyzer_results=analyzer_results)  # type: ignore[arg-type]
                cleaned.append(Document(content=anonymized.text, meta=doc.meta.copy()))
            except Exception as e:
                logger.warning(
                    "Could not anonymize document {doc_id}. Skipping it. Error: {error}",
                    doc_id=doc.id,
                    error=e,
                )
                cleaned.append(doc)
        return {"documents": cleaned}
