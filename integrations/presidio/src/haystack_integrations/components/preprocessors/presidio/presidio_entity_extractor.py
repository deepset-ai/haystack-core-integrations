# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack import Document, component, logging
from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)


@component
class PresidioEntityExtractor:
    """
    Detects PII entities in Haystack Documents using [Microsoft Presidio Analyzer](https://microsoft.github.io/presidio/).

    Accepts a list of Documents and returns new Documents with detected PII entities stored
    in each Document's metadata under the key `"entities"`. Each entry in the list contains
    the entity type, start/end character offsets, and the confidence score.

    Original Documents are not mutated. Documents without text content are passed through unchanged.

    Call `warm_up()` before running this component to load the Presidio analyzer engine.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.preprocessors.presidio import PresidioEntityExtractor

    extractor = PresidioEntityExtractor()
    extractor.warm_up()
    result = extractor.run(documents=[Document(content="Contact Alice at alice@example.com")])
    print(result["documents"][0].meta["entities"])
    # [{"entity_type": "PERSON", "start": 8, "end": 13, "score": 0.85},
    #  {"entity_type": "EMAIL_ADDRESS", "start": 17, "end": 34, "score": 1.0}]
    ```
    """

    def __init__(
        self,
        *,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
    ) -> None:
        """
        Initializes the PresidioEntityExtractor.

        :param language:
            Language code for PII detection. Defaults to `"en"`.
            See [Presidio supported languages](https://microsoft.github.io/presidio/supported_languages/).
        :param entities:
            List of PII entity types to detect (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
            If `None`, all supported entity types are detected.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be included. Defaults to `0.35`.
            See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self._analyzer: AnalyzerEngine | None = None

    def warm_up(self) -> None:
        """
        Initializes the Presidio analyzer engine.

        This method loads the underlying NLP models and should be called before `run()`.
        In a Haystack Pipeline, this is called automatically before the first run.
        """
        if self._analyzer is None:
            self._analyzer = AnalyzerEngine()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Detects PII entities in the provided Documents.

        :param documents:
            List of Documents to analyze for PII entities.
        :returns:
            A dictionary with key `documents` containing Documents with detected entities
            stored in metadata under the key `"entities"`.
        """
        result_docs: list[Document] = []
        for doc in documents:
            if doc.content is None:
                result_docs.append(doc)
                continue
            if self._analyzer is None:
                msg = "The component was not warmed up. Call warm_up() before running it."
                raise RuntimeError(msg)
            try:
                analyzer_results = self._analyzer.analyze(
                    text=doc.content,
                    language=self.language,
                    entities=self.entities,
                    score_threshold=self.score_threshold,
                )
                entities = [
                    {
                        "entity_type": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score,
                    }
                    for r in analyzer_results
                ]
                result_docs.append(replace(doc, meta={**doc.meta, "entities": entities}))
            except Exception as e:
                logger.warning(
                    "Could not extract entities from document {doc_id}. Skipping it. Error: {error}",
                    doc_id=doc.id,
                    error=e,
                )
                result_docs.append(doc)
        return {"documents": result_docs}
