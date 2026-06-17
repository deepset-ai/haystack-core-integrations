# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.utils.device import ComponentDevice
from thinc.api import get_current_ops, set_current_ops

import spacy
from spacy import Language as SpacyPipeline


@dataclass
class NamedEntityAnnotation:
    """
    Describes a single NER annotation.

    :param entity:
        Entity label.
    :param start:
        Start index of the entity in the document.
    :param end:
        End index of the entity in the document.
    :param score:
        Score calculated by the model.
    """

    entity: str
    start: int
    end: int
    score: float | None = None


@component
class SpacyNamedEntityExtractor:
    """
    Annotates named entities in a collection of documents.

    The component can be used with any [spaCy model](https://spacy.io/models) that contains
    an NER component. Annotations are stored as metadata in the documents.

    Usage example:
    ```python
    from haystack import Document

    from haystack_integrations.components.extractors.spacy import SpacyNamedEntityExtractor

    documents = [
        Document(content="I'm Merlin, the happy pig!"),
        Document(content="My name is Clara and I live in Berkeley, California."),
    ]
    extractor = SpacyNamedEntityExtractor(model="en_core_web_sm")
    results = extractor.run(documents=documents)["documents"]
    annotations = [SpacyNamedEntityExtractor.get_stored_annotations(doc) for doc in results]
    print(annotations)
    ```
    """

    _METADATA_KEY = "named_entities"

    def __init__(
        self,
        *,
        model: str,
        pipeline_kwargs: dict[str, Any] | None = None,
        device: ComponentDevice | None = None,
    ) -> None:
        """
        Create a Named Entity extractor component.

        :param model:
            Name of the spaCy model or a path to the model on
            the local disk.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        :param device:
            The device on which the model is loaded. If `None`,
            the default device is automatically selected.
        :raises ValueError:
            If the device represents multiple devices, which the
            spaCy backend does not support.
        """
        self.model_name_or_path = model
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.device = ComponentDevice.resolve_device(device)

        if self.device.has_multiple_devices:
            msg = "spaCy backend for named entity extractor only supports inference on single devices"
            raise ValueError(msg)

        self.pipeline: SpacyPipeline | None = None
        self._warmed_up: bool = False

    def warm_up(self) -> None:
        """
        Initialize the component.

        :raises ComponentError:
            If the component fails to initialize successfully.
        """
        if self._warmed_up:
            return

        try:
            # We need to initialize the model on the GPU if needed.
            with self._select_device():
                self.pipeline = spacy.load(self.model_name_or_path)

            if not self.pipeline.has_pipe("ner"):
                msg = f"spaCy pipeline '{self.model_name_or_path}' does not contain an NER component"
                raise ComponentError(msg)

            # Disable unnecessary pipes.
            pipes_to_keep = ("ner", "tok2vec", "transformer", "curated_transformer")
            for name in self.pipeline.pipe_names:
                if name not in pipes_to_keep:
                    self.pipeline.disable_pipe(name)

            self.pipeline_kwargs = {k: v for k, v in self.pipeline_kwargs.items() if k not in ("texts", "batch_size")}
            self._warmed_up = True
        except Exception as e:
            msg = f"{self.__class__.__name__} failed to initialize."
            raise ComponentError(msg) from e

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], batch_size: int = 1) -> dict[str, Any]:
        """
        Annotate named entities in each document and store the annotations in the document's metadata.

        :param documents:
            Documents to process.
        :param batch_size:
            Batch size used for processing the documents.
        :returns:
            Processed documents.
        :raises ComponentError:
            If the model fails to process a document.
        """
        if not self._warmed_up:
            self.warm_up()

        texts = [doc.content if doc.content is not None else "" for doc in documents]
        annotations = self._annotate(texts, batch_size=batch_size)

        if len(annotations) != len(documents):
            msg = (
                "NER model did not return the correct number of annotations; "
                f"got {len(annotations)} but expected {len(documents)}"
            )
            raise ComponentError(msg)

        new_documents = []
        for doc, doc_annotations in zip(documents, annotations, strict=True):
            new_meta = {**doc.meta, self._METADATA_KEY: doc_annotations}
            new_documents.append(replace(doc, meta=new_meta))

        return {"documents": new_documents}

    def _annotate(self, texts: list[str], *, batch_size: int = 1) -> list[list[NamedEntityAnnotation]]:
        """
        Predict annotations for a collection of documents.

        :param texts:
            Raw texts to be annotated.
        :param batch_size:
            Size of text batches that are
            passed to the model.
        :returns:
            NER annotations.
        """
        if not self.initialized:
            msg = "NER model was not initialized - Did you call `warm_up()`?"
            raise ComponentError(msg)

        assert self.pipeline is not None  # noqa: S101
        with self._select_device():
            outputs = list(self.pipeline.pipe(texts=texts, batch_size=batch_size, **self.pipeline_kwargs))

        return [
            [
                NamedEntityAnnotation(entity=entity.label_, start=entity.start_char, end=entity.end_char)
                for entity in doc.ents
            ]
            for doc in outputs
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name_or_path,
            device=self.device,
            pipeline_kwargs=self.pipeline_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpacyNamedEntityExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        try:
            return default_from_dict(cls, data)
        except Exception as e:
            msg = f"Couldn't deserialize {cls.__name__} instance"
            raise DeserializationError(msg) from e

    @property
    def initialized(self) -> bool:
        """
        Returns if the extractor is ready to annotate text.
        """
        return self.pipeline is not None

    @classmethod
    def get_stored_annotations(cls, document: Document) -> list[NamedEntityAnnotation] | None:
        """
        Returns the document's named entity annotations stored in its metadata, if any.

        :param document:
            Document whose annotations are to be fetched.
        :returns:
            The stored annotations.
        """

        return document.meta.get(cls._METADATA_KEY)

    @contextmanager
    def _select_device(self) -> Iterator[None]:
        """
        Context manager used to run spaCy models on a specific GPU in a scoped manner.
        """
        device_id = self.device.to_spacy()
        previous_ops = get_current_ops()
        try:
            if device_id >= 0:
                spacy.require_gpu(device_id)
            yield
        finally:
            set_current_ops(previous_ops)
