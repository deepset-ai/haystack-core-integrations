# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Optional
from urllib.parse import urlparse, urlunparse

from .client import Client
from .models import MODEL_TABLE, Model


def url_validation(api_url: str) -> str:
    """
    Validate and normalize an API URL.

    :param api_url:
        The API URL to validate and normalize.
    :returns:
        A normalized version of the API URL with '/v1' path appended, if needed.
    :raises ValueError:
        If the base URL path is not recognized or does not match expected format.
    """
    if api_url is not None:
        parsed = urlparse(api_url)

        # Ensure scheme and netloc (domain name) are present
        if not (parsed.scheme and parsed.netloc):
            expected_format = "Expected format is: http://host:port"
            msg = f"Invalid api_url format. {expected_format} Got: {api_url}"
            raise ValueError(msg)

        normalized_path = parsed.path.rstrip("/")
        if not normalized_path.endswith("/v1"):
            warnings.warn(f"{api_url} does not end in /v1, you may have inference and listing issues", stacklevel=2)
            normalized_path += "/v1"

            api_url = urlunparse((parsed.scheme, parsed.netloc, normalized_path, None, None, None))
    return api_url


def is_hosted(api_url: str) -> bool:
    """Check if the api_url belongs to api catalogue."""
    return urlparse(api_url).netloc in [
        "integrate.api.nvidia.com",
        "ai.api.nvidia.com",
    ]


def lookup_model(name: str) -> Optional[Model]:
    """
    Lookup a model by name, using only the table of known models.
    The name is either:
        - directly in the table
        - an alias in the table
        - not found (None)
    Callers can check to see if the name was an alias by
    comparing the result's id field to the name they provided.
    """
    if not (model := MODEL_TABLE.get(name)):
        for mdl in MODEL_TABLE.values():
            if mdl.aliases and name in mdl.aliases:
                model = mdl
                break
    return model


def determine_model(name: str) -> Optional[Model]:
    """
    Determine the model to use based on a name, using
    only the table of known models.

    Raise a warning if the model is found to be
    an alias of a known model.

    If the model is not found, return None.
    """
    if model := lookup_model(name):
        # all aliases are deprecated
        if model.id != name:
            warn_msg = f"Model {name} is deprecated. Using {model.id} instead."
            warnings.warn(warn_msg, UserWarning, stacklevel=1)
    return model


def validate_hosted_model(
    model_name: str,
    client: Optional[Client] = None,
) -> Model:
    """
    Checks if a given model is compatible with given client.

    :param model_name: The name of the model.
    :param client: client name enum, e.g. Client.NVIDIA_GENERATOR, Client.NVIDIA_RANKER,
                   Client.NVIDIA_TEXT_EMBEDDER, Client.NVIDIA_DOCUMENT_EMBEDDER

    :raises ValueError: If the model is incompatible with the client or if the model is unknown.

    :returns: The model if it is compatible with the client.
    """
    supported = {
        Client.NVIDIA_GENERATOR: ("chat",),
        Client.NVIDIA_TEXT_EMBEDDER: ("embedding",),
        Client.NVIDIA_DOCUMENT_EMBEDDER: ("embedding",),
        Client.NVIDIA_RANKER: ("ranking",),
    }

    if model := determine_model(model_name):
        err_msg = f"Model {model.id} is incompatible with client {client}. Please check `{client}.available_models`."

        if client and model.client and model.client != client:
            raise ValueError(err_msg)
        elif client and not model.client and model.model_type not in supported[client]:
            raise ValueError(err_msg)
        elif not client:
            warn_msg = f"Unable to determine validity of {model.id}"
            warnings.warn(warn_msg, stacklevel=1)

    else:
        err_msg = f"Model {model_name} is unknown, check `available_models`"
        raise ValueError(err_msg)
    return model
