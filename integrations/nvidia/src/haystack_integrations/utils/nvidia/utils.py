import warnings
from typing import List
from urllib.parse import urlparse, urlunparse


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
