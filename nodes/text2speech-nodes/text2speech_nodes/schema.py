# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional, Dict

from pathlib import Path
from dataclasses import asdict

from pydantic.dataclasses import dataclass  # See haystack.schema import statements

from haystack.schema import Document, Answer


@dataclass
class SpeechDocument(Document):
    """
    Text-based document that also contains some accessory audio information
    (either generated from the text with text to speech nodes, or extracted
    from an audio source containing spoken words).

    Note: for documents of this type the primary information source is *text*,
    so this is _not_ an audio document. The embeddings are computed on the textual
    representation and will work with regular, text-based nodes and pipelines.
    """

    content_audio: Optional[Path] = None  # FIXME this should be mandatory, fix the pydantic hierarchy to allow it.

    def __repr__(self):
        return f"<SpeechDocument: {str(self.to_dict())}>"

    def __str__(self):
        # In some cases, self.content is None (therefore not subscriptable)
        if self.content is None:
            return f"<SpeechDocument: id={self.id}, content=None>"
        return (
            f"<SpeechDocument: id={self.id}, content='{self.content[:100]} "
            f"{'...' if len(self.content) > 100 else ''}', content_audio={self.content_audio}>"
        )

    def to_dict(self, field_map={}) -> Dict:
        dictionary = super().to_dict(field_map=field_map)
        for key, value in dictionary.items():
            if isinstance(value, Path):
                dictionary[key] = str(value.absolute())
        return dictionary

    @classmethod
    def from_dict(cls, dict, field_map={}, id_hash_keys=None):
        doc = super().from_dict(dict=dict, field_map=field_map, id_hash_keys=id_hash_keys)
        doc.content_audio = Path(dict["content_audio"])
        return doc

    @classmethod
    def from_text_document(
        cls, document_object: Document, audio_content: Any = None, additional_meta: Optional[Dict[str, Any]] = None
    ):
        doc_dict = document_object.to_dict()
        doc_dict = {key: value for key, value in doc_dict.items() if value}

        doc_dict["content_audio"] = audio_content
        doc_dict["content_type"] = "audio"
        doc_dict["meta"] = {**(document_object.meta or {}), **(additional_meta or {})}

        return cls(**doc_dict)


@dataclass
class SpeechAnswer(Answer):
    """
    Text-based answer that also contains some accessory audio information
    (either generated from the text with text to speech nodes, or extracted
    from an audio source containing spoken words).

    Note: for answer of this type the primary information source is *text*,
    so this is _not_ an audio document. The embeddings are computed on the textual
    representation and will work with regular, text-based nodes and pipelines.
    """

    answer_audio: Optional[Path] = None  # FIXME this should be mandatory, fix the pydantic hierarchy to allow it.
    context_audio: Optional[Path] = None  # FIXME this should be mandatory, fix the pydantic hierarchy to allow it.

    def __str__(self):
        # self.context might be None (therefore not subscriptable)
        if not self.context:
            return (
                f"<SpeechAnswer: answer='{self.answer}', answer_audio={self.answer_audio}, "
                f"score={self.score}, context=None>"
            )
        return (
            f"<SpeechAnswer: answer='{self.answer}', answer_audio={self.answer_audio}, score={self.score}, "
            f"context='{self.context[:50]}{'...' if len(self.context) > 50 else ''}', "
            f"context_audio={self.context_audio}>"
        )

    def __repr__(self):
        return f"<SpeechAnswer {asdict(self)}>"

    def to_dict(self):
        dictionary = super().to_dict()
        for key, value in dictionary.items():
            if isinstance(value, Path):
                dictionary[key] = str(value.absolute())
        return dictionary

    @classmethod
    def from_dict(cls, dict: dict):
        for key, value in dict.items():
            if key in ["answer_audio", "context_audio"]:
                dict[key] = Path(value)
        return super().from_dict(dict=dict)

    @classmethod
    def from_text_answer(
        cls,
        answer_object: Answer,
        audio_answer: Any,
        audio_context: Optional[Any] = None,
        additional_meta: Optional[Dict[str, Any]] = None,
    ):
        answer_dict = answer_object.to_dict()
        answer_dict = {key: value for key, value in answer_dict.items() if value}

        answer_dict["answer_audio"] = audio_answer
        if audio_context:
            answer_dict["context_audio"] = audio_context
        answer_dict["meta"] = {**(answer_object.meta or {}), **(additional_meta or {})}

        return cls(**answer_dict)
