# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack import Document, component, logging
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

logger = logging.getLogger(__name__)

# Maps ISO 639-1 language codes to the largest available spaCy model for that language.
# Used to automatically configure the NLP engine when only `language` is specified.
# See https://spacy.io/models for the full list of available models.
SPACY_DEFAULT_MODELS: dict[str, str] = {
    "ca": "ca_core_news_lg",
    "zh": "zh_core_web_lg",
    "hr": "hr_core_news_lg",
    "da": "da_core_news_lg",
    "nl": "nl_core_news_lg",
    "en": "en_core_web_lg",
    "fi": "fi_core_news_lg",
    "fr": "fr_core_news_lg",
    "de": "de_core_news_lg",
    "el": "el_core_news_lg",
    "it": "it_core_news_lg",
    "ja": "ja_core_news_lg",
    "ko": "ko_core_news_lg",
    "lt": "lt_core_news_lg",
    "mk": "mk_core_news_lg",
    "nb": "nb_core_news_lg",
    "pl": "pl_core_news_lg",
    "pt": "pt_core_news_lg",
    "ro": "ro_core_news_lg",
    "ru": "ru_core_news_lg",
    "sl": "sl_core_news_lg",
    "es": "es_core_news_lg",
    "sv": "sv_core_news_lg",
    "uk": "uk_core_news_lg",
}


@component
class PresidioEntityExtractor:
    """
    Detects PII entities in Haystack Documents using Microsoft Presidio Analyzer.

    See [Presidio Analyzer](https://microsoft.github.io/presidio/) for details.

    Accepts a list of Documents and returns new Documents with detected PII entities stored
    in each Document's metadata under the key `"entities"`. Each entry in the list contains
    the entity type, start/end character offsets, and the confidence score.

    Original Documents are not mutated. Documents without text content are passed through unchanged.

    The analyzer engine is loaded on the first call to `run()`,
    or by calling `warm_up()` explicitly beforehand.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.extractors.presidio import PresidioEntityExtractor

    extractor = PresidioEntityExtractor()
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
        models: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Initializes the PresidioEntityExtractor.

        :param language:
            ISO 639-1 language code for PII detection. Defaults to `"en"`.
            For languages in the built-in mapping (e.g. `"de"`, `"fr"`, `"es"`), the appropriate
            spaCy model is loaded automatically at warm-up time — no need to set `models`.
            For unsupported languages, use the `models` parameter to configure a custom model.
            See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
        :param entities:
            List of PII entity types to detect (e.g. `["PERSON", "EMAIL_ADDRESS"]`).
            If `None`, all supported entity types are detected.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be included. Defaults to `0.35`.
            See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
        :param models:
            Advanced override: list of spaCy model configurations.
            Each entry must contain `"lang_code"` and `"model_name"` keys,
            e.g. `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`.
            Use this only when you need a specific model variant or a language not covered by the
            built-in mapping. If `None`, the model is selected automatically from `SPACY_DEFAULT_MODELS`
            based on `language`.
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self.models = models
        self._analyzer: AnalyzerEngine | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Initializes the Presidio analyzer engine.

        This method loads the underlying NLP models. In a Haystack Pipeline,
        this is called automatically before the first run.
        """
        if self._is_warmed_up:
            return

        models = self.models
        if models is None:
            if self.language not in SPACY_DEFAULT_MODELS:
                supported = ", ".join(sorted(SPACY_DEFAULT_MODELS))
                msg = (
                    f"No default spaCy model is available for language '{self.language}'. "
                    f"Use the `models` parameter to specify a custom model. "
                    f"Languages with built-in support: {supported}."
                )
                raise ValueError(msg)
            models = [{"lang_code": self.language, "model_name": SPACY_DEFAULT_MODELS[self.language]}]

        nlp_engine = NlpEngineProvider(
            nlp_configuration={"nlp_engine_name": "spacy", "models": models}
        ).create_engine()
        supported_languages = [m["lang_code"] for m in models]
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=supported_languages)

        self._is_warmed_up = True

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
        if not self._is_warmed_up:
            self.warm_up()

        result_docs: list[Document] = []
        for doc in documents:
            if doc.content is None:
                result_docs.append(doc)
                continue
            try:
                analyzer_results = self._analyzer.analyze(  # type: ignore[union-attr]
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
                    "Could not extract entities from {doc_id}. Skipping extraction, keeping document. Error: {error}",
                    doc_id=doc.id,
                    error=e,
                )
                result_docs.append(doc)
        return {"documents": result_docs}
