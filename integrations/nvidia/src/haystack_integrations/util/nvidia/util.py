from urllib.parse import urlparse


def is_hosted(api_url: str):
    """"""
    return urlparse(api_url).netloc in [
        "integrate.api.nvidia.com",
        "ai.api.nvidia.com",
    ]
