# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

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
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
    ) -> None:
        """
        Initializes the PresidioTextCleaner.

        :param language:
            Language code for PII detection. Defaults to `"en"`.
        :param entities:
            List of PII entity types to detect and anonymize (e.g. `["PERSON", "PHONE_NUMBER"]`).
            If `None`, all supported entity types are used.
        :param score_threshold:
            Minimum confidence score (0-1) for a detected entity to be anonymized. Defaults to `0.35`.
        """
        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    @component.output_types(texts=list[str])
    def run(self, texts: list[str]) -> dict[str, Any]:
        """
        Anonymizes PII in the provided strings.

        :param texts:
            List of strings to anonymize.
        :returns:
            A dictionary with key `texts` containing the cleaned strings.
        """
        cleaned: list[str] = []
        for text in texts:
            try:
                analyzer_results = self._analyzer.analyze(
                    text=text,
                    language=self.language,
                    entities=self.entities,
                    score_threshold=self.score_threshold,
                )
                anonymized = self._anonymizer.anonymize(text=text, analyzer_results=analyzer_results)  # type: ignore[arg-type]
                cleaned.append(anonymized.text)
            except Exception as e:
                logger.warning(
                    "Could not anonymize text. Skipping it. Error: {error}",
                    error=e,
                )
                cleaned.append(text)
        return {"texts": cleaned}
