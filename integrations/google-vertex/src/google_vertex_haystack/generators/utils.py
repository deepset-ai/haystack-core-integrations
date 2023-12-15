import os
from typing import Optional

import vertexai
from google.auth.api_key import Credentials


def authenticate(api_key: str = "", project_id: str = "", location: Optional[str] = None):
    """
    Authenticates using the provided API key, project ID, and location.

    If api_key is not provided, it will use the GOOGLE_API_KEY environment variable.
    If neither are set, will attempt to use Application Default Credentials (ADCs).
    If no ADC is not set, will raise an error.
    For more information on ADC see the official Google documentation:
        https://cloud.google.com/docs/authentication/provide-credentials-adc
    """
    credentials = None
    if not api_key and "GOOGLE_API_KEY" in os.environ:
        Credentials(token=os.environ["GOOGLE_API_KEY"])
    vertexai.init(project=project_id, location=location, credentials=credentials)
