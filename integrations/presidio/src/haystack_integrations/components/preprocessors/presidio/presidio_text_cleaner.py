# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import component, logging
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

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

    def __init__(
        self,
        *,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
    ) -> None:
        """
        Initializes the PresidioTextCleaner.

        :param language:
            Language code for PII detection. Defaults to `"en"`.
            See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
        :param entities:
            List of PII entity types to detect and anonymize (e.g. `["PERSON", "PHONE_NUMBER"]`).
            If `None`, all supported entity types are used.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
            See [Presidio analyzer documentation](https://microsoft.github.io/presidio/analyzer/).
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
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

        self._analyzer = AnalyzerEngine()
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
