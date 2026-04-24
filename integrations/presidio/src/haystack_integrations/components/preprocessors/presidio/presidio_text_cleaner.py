# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from haystack import component, logging
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from haystack_integrations.components.common.presidio.utils import SPACY_DEFAULT_MODELS as _SPACY_DEFAULT_MODELS

logger = logging.getLogger(__name__)


@component
class PresidioTextCleaner:
    """
    Anonymizes PII in plain strings using [Microsoft Presidio](https://microsoft.github.io/presidio/).

    Accepts a list of strings, detects personally identifiable information (PII), and returns
    a new list of strings with PII replaced by entity type placeholders (e.g. `<PERSON>`).
    Useful for sanitizing user queries before they are sent to an LLM.

    The analyzer and anonymizer engines are loaded on the first call to `run()`,
    or by calling `warm_up()` explicitly beforehand.

    ### Usage example

    ```python
    from haystack_integrations.components.preprocessors.presidio import PresidioTextCleaner

    cleaner = PresidioTextCleaner()
    result = cleaner.run(texts=["Hi, I am John Smith, call me at 212-555-1234"])
    print(result["texts"][0])
    # Hi, I am <PERSON>, call me at <PHONE_NUMBER>
    ```
    """

    SPACY_DEFAULT_MODELS: ClassVar[dict[str, str]] = _SPACY_DEFAULT_MODELS
    """Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

    Used to automatically select an NLP model when `models` is not specified.
    See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.
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
        Initializes the PresidioTextCleaner.

        :param language:
            ISO 639-1 language code for PII detection. Defaults to `"en"`.
            For languages in the built-in mapping (e.g. `"de"`, `"fr"`, `"es"`), the appropriate
            spaCy model is loaded automatically at warm-up time — no need to set `models`.
            For unsupported languages, use the `models` parameter to configure a custom model.
            See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
        :param entities:
            List of PII entity types to detect and anonymize (e.g. `["PERSON", "PHONE_NUMBER"]`).
            If `None`, all supported entity types are used.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
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

        models = self.models
        if models is None:
            if self.language not in self.SPACY_DEFAULT_MODELS:
                supported = ", ".join(sorted(self.SPACY_DEFAULT_MODELS))
                msg = (
                    f"No default spaCy model is available for language '{self.language}'. "
                    f"Use the `models` parameter to specify a custom model. "
                    f"Languages with built-in support: {supported}."
                )
                raise ValueError(msg)
            models = [{"lang_code": self.language, "model_name": self.SPACY_DEFAULT_MODELS[self.language]}]

        nlp_engine = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": models}).create_engine()
        supported_languages = [m["lang_code"] for m in models]
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=supported_languages)
        self._anonymizer = AnonymizerEngine()

        self._is_warmed_up = True

    @component.output_types(texts=list[str])
    def run(self, texts: list[str]) -> dict[str, list[str]]:
        """
        Anonymizes PII in the provided strings.

        :param texts:
            List of strings to anonymize.
        :returns:
            A dictionary with key `texts` containing the cleaned strings.
        """
        if not self._is_warmed_up:
            self.warm_up()

        cleaned: list[str] = []
        for text in texts:
            try:
                analyzer_results = self._analyzer.analyze(  # type: ignore[union-attr]
                    text=text,
                    language=self.language,
                    entities=self.entities,
                    score_threshold=self.score_threshold,
                )
                anonymized = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)  # type: ignore[arg-type, union-attr]
                cleaned.append(anonymized.text)
            except Exception as e:
                logger.warning(
                    "Could not anonymize text. Skipping it. Error: {error}",
                    error=e,
                )
                cleaned.append(text)
        return {"texts": cleaned}
