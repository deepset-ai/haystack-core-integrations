import warnings
from urllib.parse import urlparse, urlunparse


def url_validation(api_url: str, default_api_url: str, allowed_paths: list):
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
            warn_msg = f"{expected_format} Rest is ingnored."
            warnings.warn(warn_msg, stacklevel=2)
        else:
            err_msg = f"Base URL path is not recognized. {expected_format}"
            raise ValueError(err_msg)

    base_url = urlunparse((result.scheme, result.netloc, "v1", "", "", ""))
    return base_url
