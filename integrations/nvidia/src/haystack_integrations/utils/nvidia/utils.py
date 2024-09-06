import warnings
from typing import List, Optional
from urllib.parse import urlparse, urlunparse

from .statics import MODEL_TABLE, Model


def url_validation(api_url: str, default_api_url: str, allowed_paths: List[str]) -> str:
    """
    Validate and normalize an API URL.

    :param api_url:
        The API URL to validate and normalize.
    :param default_api_url:
        The default API URL for comparison.
    :param allowed_paths:
        A list of allowed base paths that are valid if present in the URL.
    :returns:
        A normalized version of the API URL with '/v1' path appended, if needed.
    :raises ValueError:
        If the base URL path is not recognized or does not match expected format.
    """
    ## Making sure /v1 in added to the url, followed by infer_path
    result = urlparse(api_url)
    expected_format = "Expected format is 'http://host:port'."

    if api_url == default_api_url:
        return api_url
    if result.path:
        normalized_path = result.path.strip("/")
        if normalized_path == "v1":
            pass
        elif normalized_path in allowed_paths:
            warn_msg = f"{expected_format} Rest is ignored."
            warnings.warn(warn_msg, stacklevel=2)
        else:
            err_msg = f"Base URL path is not recognized. {expected_format}"
            raise ValueError(err_msg)

    base_url = urlunparse((result.scheme, result.netloc, "v1", "", "", ""))
    return base_url


def is_hosted(api_url: str):
    """"""
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
    model = None
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


def validate_hosted_model(class_name: str, model_name: str, client) -> None:
    """
    Validates compatibility of the hosted model with the client.

    Args:
        model_name (str): The name of the model.

    Raises:
        ValueError: If the model is incompatible with the client.
    """
    if model := determine_model(model_name):
        if not model.client:
            warn_msg = f"Unable to determine validity of {model.id}"
            warnings.warn(warn_msg, stacklevel=1)
        elif model.model_type == "embedding" and class_name in ["NvidiaTextEmbedder", "NvidiaDocumentEmbedder"]:
            pass
        elif model.client != class_name:
            err_msg = f"Model {model.id} is incompatible with client {class_name}. \
                        Please check `{class_name}.available_models`."
            raise ValueError(err_msg)
    else:
        candidates = [model for model in client.available_models if model.id == model_name]
        assert len(candidates) <= 1, f"Multiple candidates for {model_name} in `available_models`: {candidates}"
        if candidates:
            model = candidates[0]
            warn_msg = f"Found {model_name} in available_models, but type is unknown and inference may fail."
            warnings.warn(warn_msg, stacklevel=1)
        else:
            err_msg = f"Model {model_name} is unknown, check `available_models`"
            raise ValueError(err_msg)
    # else:
    #     if model_name not in [model.id for model in client.available_models]:
    #         raise ValueError(f"No locally hosted {model_name} was found.")
