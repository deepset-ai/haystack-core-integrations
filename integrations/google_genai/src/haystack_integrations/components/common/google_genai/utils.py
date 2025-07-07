# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

from google.genai import Client
from haystack import logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)


def _get_client(
    api_key: Secret,
    api: Literal["gemini", "vertex"],
    vertex_ai_project: Optional[str],
    vertex_ai_location: Optional[str],
) -> Client:
    """
    Internal utility function to get a Google GenAI client.

    Supports:
    - Gemini Developer API (API Key Authentication)
    - Vertex AI (Application Default Credentials)
    - Vertex AI (API Key Authentication).

    :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
    :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
    :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
        Application Default Credentials.
    :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1"). Required
        when using Vertex AI with Application Default Credentials.

    :returns: A Google GenAI client.

    :raises: ValueError if Gemini API is used without providing an API key or if Vertex AI is used without providing
        an API key or both vertex_ai_project and vertex_ai_location.
    """

    if api not in ["gemini", "vertex"]:
        msg = f"Invalid API: {api}. Must be either 'gemini' or 'vertex'."
        raise ValueError(msg)

    resolved_api_key = api_key.resolve_value()

    if api == "vertex":
        if not resolved_api_key and not (vertex_ai_project and vertex_ai_location):
            msg = (
                "To use Vertex AI, you must provide both vertex_ai_project and vertex_ai_location or export "
                "the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
            raise ValueError(msg)

        if vertex_ai_project and vertex_ai_location:
            logger.info("Using vertex_ai_project and vertex_ai_location for authentication.")
            return Client(vertexai=True, project=vertex_ai_project, location=vertex_ai_location)

        logger.info(
            "No vertex_ai_project or vertex_ai_location provided for Vertex AI. Using the API key for authentication."
        )
        return Client(vertexai=True, api_key=resolved_api_key)

    # Gemini API
    if not resolved_api_key:
        msg = "To use Gemini API, you must export the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        raise ValueError(msg)

    return Client(api_key=resolved_api_key)
