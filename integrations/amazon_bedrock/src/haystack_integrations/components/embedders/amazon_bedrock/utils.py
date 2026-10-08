# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, get_args

EmbeddingModelFamily = Literal["cohere", "titan"]


def _resolve_model_family(model: str, model_family: EmbeddingModelFamily | None) -> EmbeddingModelFamily:
    """
    Resolve the model family that determines the request and response format for `model`.

    An explicit `model_family` takes precedence. Otherwise the family is detected from the model ID, which only works
    when the ID names the model. Application inference profile ARNs, for example, don't.

    :param model: The Amazon Bedrock model ID or ARN.
    :param model_family: The model family set by the user.
    :returns: The model family.
    :raises ValueError: If `model_family` is not one of the supported families, or if it's `None` and `model` contains
        neither "cohere" nor "titan".
    """
    if model_family is not None:
        # `model_family` comes from user input or a serialized pipeline, so the Literal type isn't enforced
        if model_family not in get_args(EmbeddingModelFamily):
            msg = f"Model family {model_family!r} is not supported. Use one of {get_args(EmbeddingModelFamily)}."
            raise ValueError(msg)
        return model_family

    if "cohere" in model:
        return "cohere"
    if "titan" in model:
        return "titan"

    msg = (
        f"Model {model} is not supported. Only Amazon Titan and Cohere embedding models are supported. "
        "If the model ID doesn't name the model, as with application inference profile ARNs, "
        "set `model_family` to 'cohere' or 'titan'."
    )
    raise ValueError(msg)


def _supports_titan_v2_params(model: str) -> bool:
    """
    Check whether `dimensions` and `normalize` can be sent to the Amazon Titan `model`.

    Only Amazon Titan Text Embeddings V2 supports them. An ID that doesn't name a Titan embedding model, such as an
    application inference profile ARN, can't be checked, so the parameters are sent as configured.

    :param model: The Amazon Bedrock model ID or ARN.
    :returns: `False` if `model` names a Titan embedding model other than Text Embeddings V2, `True` otherwise.
    """
    return "amazon.titan-embed-text-v2" in model or "amazon.titan-embed" not in model
