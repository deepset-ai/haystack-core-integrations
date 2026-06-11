# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
