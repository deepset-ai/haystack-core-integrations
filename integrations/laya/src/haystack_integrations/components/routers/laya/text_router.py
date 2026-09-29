# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

from laya.agent import Agent, load

_LOW_CONFIDENCE = "low_confidence"


@component
class LayaTextRouter:
    """
    Routes a text to the connection of the label a Laya decision model picks for it.

    Laya answers a single `choice` question over `labels` in one forward pass and returns calibrated probabilities.
    The component has one output per label. When `min_confidence` is set, it also has a `low_confidence` output
    that receives the text whenever the answer's confidence is below the threshold, so uncertain texts can go to a
    fallback such as an LLM or a human. See the [model card](https://huggingface.co/convaiinnovations/laya) for the
    available checkpoints.

    ### Usage example

    ```python
    from haystack_integrations.components.routers.laya import LayaTextRouter

    router = LayaTextRouter(
        labels={
            "billing": "Payments, invoices, refunds and charges",
            "technical": "Bugs, crashes and errors in the product",
        },
        instructions="Which team should handle this support request?",
        min_confidence=0.6,
    )

    print(router.run(text="I was charged twice for my subscription."))
    # {'billing': 'I was charged twice for my subscription.'}
    ```
    """

    def __init__(
        self,
        labels: list[str] | dict[str, str],
        *,
        instructions: str = "Which category does this text belong to?",
        model: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        revision: str | None = None,
        device: ComponentDevice | None = None,
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        min_confidence: float | None = None,
    ) -> None:
        """
        Creates a LayaTextRouter.

        :param labels:
            The labels to route between, each becoming an output connection. Pass a dict of label to description to
            tell the model what each label means.
        :param instructions:
            The question the model answers by picking one of the labels.
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
        :param min_confidence:
            Threshold between 0 and 1 for the probability of the picked label (`answer_confidence`). Texts below it
            are sent to the `low_confidence` output. If `None`, every text goes to a label output.
        :raises ValueError:
            If `labels` is empty or contains `low_confidence`, or if `min_confidence` is outside 0 to 1.
        """
        if not labels:
            msg = "`labels` must contain at least one label."
            raise ValueError(msg)
        if _LOW_CONFIDENCE in labels:
            msg = f"`{_LOW_CONFIDENCE}` is reserved for the low confidence output and can't be used as a label."
            raise ValueError(msg)
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            msg = f"`min_confidence` must be between 0 and 1, got {min_confidence}."
            raise ValueError(msg)

        self.labels = labels
        self.instructions = instructions
        self.model = model
        self.subfolder = subfolder
        self.revision = revision
        self.device = device
        self.token = token
        self.min_confidence = min_confidence
        self._agent: Agent | None = None

        outputs = list(labels)
        if min_confidence is not None:
            outputs.append(_LOW_CONFIDENCE)
        component.set_output_types(self, **dict.fromkeys(outputs, str))

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
            labels=self.labels,
            instructions=self.instructions,
            model=self.model,
            subfolder=self.subfolder,
            revision=self.revision,
            device=self.device.to_dict() if self.device else None,
            token=self.token.to_dict() if self.token else None,
            min_confidence=self.min_confidence,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LayaTextRouter":
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

    def run(self, text: str) -> dict[str, str]:
        """
        Routes the text to the output of the label the model picks.

        The model is loaded on the first call.

        :param text:
            The text to route.
        :returns:
            A dictionary with a single key, the picked label or `low_confidence`, mapped to the input text.
        :raises TypeError:
            If `text` is not a string.
        """
        if not isinstance(text, str):
            msg = "LayaTextRouter expects a str as input."
            raise TypeError(msg)

        self.warm_up()
        assert self._agent is not None  # noqa: S101  # mypy: the agent is loaded by warm_up above

        question = {"type": "choice", "instructions": self.instructions, "criteria": self.labels}
        result = self._agent.predict(text, {"route": question}, min_confidence=self.min_confidence)
        answer = result["answers"]["route"]

        # Laya flags answers below `min_confidence` with `low_confidence`
        if answer.get(_LOW_CONFIDENCE):
            return {_LOW_CONFIDENCE: text}
        return {answer["choice"]: text}
