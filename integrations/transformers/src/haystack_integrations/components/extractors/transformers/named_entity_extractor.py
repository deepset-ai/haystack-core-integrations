# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

from haystack_integrations.common.transformers.utils import _resolve_hf_pipeline_kwargs
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers import Pipeline as HfPipeline


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
class TransformersNamedEntityExtractor:
    """
    Annotates named entities in a collection of documents.

    The component can be used with any token classification model from the
    [Hugging Face model hub](https://huggingface.co/models). Annotations are
    stored as metadata in the documents.

    Usage example:
    ```python
    from haystack import Document

    from haystack_integrations.components.extractors.transformers import TransformersNamedEntityExtractor

    documents = [
        Document(content="I'm Merlin, the happy pig!"),
        Document(content="My name is Clara and I live in Berkeley, California."),
    ]
    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")
    results = extractor.run(documents=documents)["documents"]
    annotations = [TransformersNamedEntityExtractor.get_stored_annotations(doc) for doc in results]
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
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
    ) -> None:
        """
        Create a Named Entity extractor component.

        :param model:
            Name of the model or a path to the model on
            the local disk.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        :param device:
            The device on which the model is loaded. If `None`,
            the default device is automatically selected. If a
            device/device map is specified in `pipeline_kwargs`,
            it overrides this parameter.
        :param token:
            The API token to download private models from Hugging Face.
        """
        self.token = token
        self.model_name_or_path = model
        self.device = ComponentDevice.resolve_device(device)

        self.pipeline_kwargs = _resolve_hf_pipeline_kwargs(
            huggingface_pipeline_kwargs=pipeline_kwargs or {},
            model=model,
            task="ner",
            supported_tasks=["ner"],
            device=self.device,
            token=token,
        )

        self.tokenizer: Any = None
        self.model: AutoModelForTokenClassification | None = None
        self.pipeline: HfPipeline | None = None
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
            token = self.pipeline_kwargs.get("token", None)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, token=token)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path, token=token)

            pipeline_params: dict[str, Any] = {
                "task": "ner",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "aggregation_strategy": "simple",
            }
            pipeline_params.update({k: v for k, v in self.pipeline_kwargs.items() if k not in pipeline_params})
            self.device.update_hf_kwargs(pipeline_params, overwrite=False)
            self.pipeline = pipeline(**pipeline_params)
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
        outputs = self.pipeline(texts, batch_size=batch_size)
        return [
            [
                NamedEntityAnnotation(
                    entity=annotation["entity"] if "entity" in annotation else annotation["entity_group"],
                    start=annotation["start"],
                    end=annotation["end"],
                    score=annotation["score"],
                )
                for annotation in annotations
            ]
            for annotations in outputs
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model_name_or_path,
            device=self.device,
            pipeline_kwargs=self.pipeline_kwargs,
            token=self.token,
        )

        hf_pipeline_kwargs = serialization_dict["init_parameters"]["pipeline_kwargs"]
        hf_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(hf_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformersNamedEntityExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        try:
            init_params = data.get("init_parameters", {})
            hf_pipeline_kwargs = init_params.get("pipeline_kwargs")
            deserialize_hf_model_kwargs(hf_pipeline_kwargs or {})
            return default_from_dict(cls, data)
        except Exception as e:
            msg = f"Couldn't deserialize {cls.__name__} instance"
            raise DeserializationError(msg) from e

    @property
    def initialized(self) -> bool:
        """
        Returns if the extractor is ready to annotate text.
        """
        return (self.tokenizer is not None and self.model is not None) or self.pipeline is not None

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
