# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

from laya.agent import Agent, load

logger = logging.getLogger(__name__)

_QUESTION_TYPES = ("choice", "score", "noul")


@component
class LayaDocumentClassifier:
    """
    Answers typed questions about each document with a Laya decision model and stores the answers in its metadata.

    Laya is a non-autoregressive decision model: instead of generating text, it answers every question in a single
    forward pass and returns calibrated probabilities. Each question has one of three types:
    - `choice`: picks one label from `criteria`, a dict of label to description or a list of labels.
    - `score`: rates the text on the ordered levels in `criteria`, a list of level descriptions.
    - `noul`: returns the probability that the yes/no question in `instructions` is true.

    Answers are stored under `meta[metadata_field]`, keyed by question ID. See the
    [model card](https://huggingface.co/convaiinnovations/laya) for the available checkpoints and the full
    question and answer format.

    ### Usage example

    ```python
    from haystack import Document

    from haystack_integrations.components.classifiers.laya import LayaDocumentClassifier

    classifier = LayaDocumentClassifier(
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": ["billing", "technical", "sales"],
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["can wait", "this week", "today"],
            },
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        }
    )

    result = classifier.run(documents=[Document(content="I was charged twice, please refund me today.")])
    print(result["documents"][0].meta["laya"])
    # {'department': {'type': 'choice', 'choice': 'billing',
    #                 'probabilities': {'billing': 0.9677, 'technical': 0.0221, 'sales': 0.0102},
    #                 'confidence': 0.8519, ...},
    #  'urgency': {'type': 'score', 'score': 1.948, 'probabilities': {'0': 0.0113, '1': 0.0295, '2': 0.9593}, ...},
    #  'refund': {'type': 'noul', 'noul': 0.9498, ...}}
    ```

    ### Routing documents by their answers

    Connect the classifier to a `MetadataRouter` and match on the nested answer fields:

    ```python
    from haystack import Document, Pipeline
    from haystack.components.routers import MetadataRouter

    from haystack_integrations.components.classifiers.laya import LayaDocumentClassifier

    pipeline = Pipeline()
    pipeline.add_component(
        "classifier",
        LayaDocumentClassifier(
            questions={
                "department": {
                    "type": "choice",
                    "instructions": "Which department should handle this ticket?",
                    "criteria": ["billing", "technical"],
                }
            }
        ),
    )
    pipeline.add_component(
        "router",
        MetadataRouter(
            rules={
                label: {"field": "meta.laya.department.choice", "operator": "==", "value": label}
                for label in ["billing", "technical"]
            }
        ),
    )
    pipeline.connect("classifier.documents", "router.documents")

    result = pipeline.run({"classifier": {"documents": [Document(content="The app crashes on startup.")]}})
    print(result["router"])
    # {'billing': [], 'technical': [Document(...)], 'unmatched': []}
    ```
    """

    def __init__(
        self,
        questions: dict[str, dict[str, Any]],
        *,
        model: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        revision: str | None = None,
        device: ComponentDevice | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        classification_field: str | None = None,
        metadata_field: str = "laya",
        batch_size: int | None = None,
        min_confidence: float | None = None,
    ) -> None:
        """
        Creates a LayaDocumentClassifier.

        :param questions:
            Questions to answer for every document, keyed by question ID. Each value is a dict with a `type`
            (`choice`, `score` or `noul`), `instructions`, and for `choice` and `score` questions, `criteria`.
        :param model:
            Hugging Face model ID or local path of a Laya checkpoint.
        :param subfolder:
            Checkpoint inside the model repository, for example `multilingual` or `typed-decisions`.
            `None` loads the checkpoint at the repository root, which is English-only.
        :param revision:
            Commit SHA, branch or tag of the model repository to download.
        :param device:
            The device to load the model on. If `None`, Laya picks CUDA, MPS, XPU or CPU, in that order.
        :param token:
            The Hugging Face token used to download the model.
        :param classification_field:
            Name of the document's metadata field to classify. If `None`, `Document.content` is classified.
        :param metadata_field:
            Name of the metadata field the answers are written to.
        :param batch_size:
            Maximum number of documents per forward pass. `None` sends all documents in one pass.
        :param min_confidence:
            Threshold between 0 and 1 for each answer's `answer_confidence`. Answers below it get
            `"low_confidence": True`.
        :raises ValueError:
            If `questions` is empty, or a question has an unknown `type` or no `instructions`.
        """
        if not questions:
            msg = "`questions` must contain at least one question."
            raise ValueError(msg)
        for question_id, question in questions.items():
            if question.get("type") not in _QUESTION_TYPES:
                msg = f"Question '{question_id}' has type {question.get('type')!r}; use one of {_QUESTION_TYPES}."
                raise ValueError(msg)
            if "instructions" not in question:
                msg = f"Question '{question_id}' has no `instructions`."
                raise ValueError(msg)

        self.questions = questions
        self.model = model
        self.subfolder = subfolder
        self.revision = revision
        self.device = device
        self.token = token
        self.classification_field = classification_field
        self.metadata_field = metadata_field
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self._agent: Agent | None = None

    def warm_up(self) -> None:
        """
        Downloads and loads the Laya model.
        """
        if self._agent is None:
            self._agent = load(
                self.model,
                device=self.device.to_torch_str() if self.device else None,
                token=self.token.resolve_value() if self.token else None,
                subfolder=self.subfolder,
                revision=self.revision,
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            questions=self.questions,
            model=self.model,
            subfolder=self.subfolder,
            revision=self.revision,
            device=self.device.to_dict() if self.device else None,
            token=self.token.to_dict() if self.token else None,
            classification_field=self.classification_field,
            metadata_field=self.metadata_field,
            batch_size=self.batch_size,
            min_confidence=self.min_confidence,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayaDocumentClassifier":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(init_params, keys=["token"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Answers the questions for each document and adds the answers to its metadata.

        The model is loaded on the first call. Documents without text to classify are returned unchanged and a
        warning is logged.

        :param documents:
            Documents to classify.
        :returns:
            A dictionary with the following key:
            - `documents`: The input documents. Each classified document has a `metadata_field` entry mapping each
              question ID to its answer.
        """
        self.warm_up()
        assert self._agent is not None  # noqa: S101  # mypy: the agent is loaded by warm_up above

        # Collect the text to classify, skipping documents that have none
        texts: list[str] = []
        indices: list[int] = []
        for index, document in enumerate(documents):
            if self.classification_field is None:
                text = document.content
            else:
                text = document.meta.get(self.classification_field)
            if not text:
                logger.warning(
                    "Document {document_id} has no text to classify and is skipped.", document_id=document.id
                )
                continue
            texts.append(text)
            indices.append(index)

        if not texts:
            return {"documents": documents}

        results = self._agent.predict_batch(
            texts, self.questions, batch_size=self.batch_size, min_confidence=self.min_confidence
        )

        # Write each result into a copy of its document
        new_documents = list(documents)
        for index, result in zip(indices, results, strict=True):
            document = documents[index]
            new_documents[index] = replace(document, meta={**document.meta, self.metadata_field: result["answers"]})
        return {"documents": new_documents}
