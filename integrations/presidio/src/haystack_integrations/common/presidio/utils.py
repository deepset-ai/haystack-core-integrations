# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

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


def build_analyzer_engine(
    *, language: str, models: list[dict[str, str]] | None, default_models: dict[str, str]
) -> AnalyzerEngine:
    """
    Build a Presidio `AnalyzerEngine` backed by spaCy, loading the NLP models.

    :param language: ISO 639-1 language code used to pick a model from `default_models` when `models` is `None`.
    :param models: Explicit spaCy model configurations, each with `"lang_code"` and `"model_name"` keys. Takes
        precedence over `default_models`.
    :param default_models: Mapping from language code to spaCy model name.
    :returns: The analyzer engine.
    :raises ValueError: If `models` is `None` and `default_models` has no model for `language`.
    """
    if models is None:
        if language not in default_models:
            supported = ", ".join(sorted(default_models))
            msg = (
                f"No default spaCy model is available for language '{language}'. "
                f"Use the `models` parameter to specify a custom model. "
                f"Languages with built-in support: {supported}."
            )
            raise ValueError(msg)
        models = [{"lang_code": language, "model_name": default_models[language]}]

    nlp_engine = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": models}).create_engine()
    supported_languages = [m["lang_code"] for m in models]
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=supported_languages)
