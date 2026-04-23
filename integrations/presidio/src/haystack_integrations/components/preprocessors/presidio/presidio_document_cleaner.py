# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, component, logging
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
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

    The analyzer and anonymizer engines are loaded on the first call to `run()`,
    or by calling `warm_up()` explicitly beforehand.

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
        *,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
        models: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Initializes the PresidioDocumentCleaner.

        :param language:
            Language code for PII detection. Defaults to `"en"`.
            Presidio's default NLP engine only includes an English spaCy model. For non-English languages,
            use the `models` parameter to specify which spaCy model to load for that language.
            See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
        :param entities:
            List of PII entity types to detect and anonymize (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
            If `None`, all supported entity types are used.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
            See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
        :param models:
            List of spaCy model configurations for language support.
            Each entry must contain `"lang_code"` and `"model_name"` keys,
            e.g. `[{"lang_code": "fr", "model_name": "fr_core_news_lg"}]`.
            The corresponding spaCy model will be loaded at warm-up time.
            If `None`, Presidio's default English model (`en_core_web_lg`) is used.
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self.models = models
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Initializes the Presidio analyzer and anonymizer engines.

        This method loads the underlying NLP models. In a Haystack Pipeline,
        this is called automatically before the first run.
        """
        if self._is_warmed_up:
            return

        if self.models:
            nlp_engine = NlpEngineProvider(
                nlp_configuration={"nlp_engine_name": "spacy", "models": self.models}
            ).create_engine()
            supported_languages = [m["lang_code"] for m in self.models]
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=supported_languages)
        else:
            self._analyzer = AnalyzerEngine(supported_languages=[self.language])
        self._anonymizer = AnonymizerEngine()

        self._is_warmed_up = True

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Anonymizes PII in the provided Documents.

        :param documents:
            List of Documents whose text content will be anonymized.
        :returns:
            A dictionary with key `documents` containing the cleaned Documents.
        """
        if not self._is_warmed_up:
            self.warm_up()

        cleaned: list[Document] = []
        for doc in documents:
            if doc.content is None:
                cleaned.append(doc)
                continue
            try:
                analyzer_results = self._analyzer.analyze(  # type: ignore[union-attr]
                    text=doc.content,
                    language=self.language,
                    entities=self.entities,
                    score_threshold=self.score_threshold,
                )
                anonymized = self._anonymizer.anonymize(text=doc.content, analyzer_results=analyzer_results)  # type: ignore[arg-type, union-attr]
                cleaned.append(Document(content=anonymized.text, meta=doc.meta.copy()))
            except Exception as e:
                logger.warning(
                    "Could not anonymize document {doc_id}. Skipping it. Error: {error}",
                    doc_id=doc.id,
                    error=e,
                )
        return {"documents": cleaned}
