# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import requests
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from .reader_mode import JinaReaderMode

READER_ENDPOINT_URL_BY_MODE = {
    JinaReaderMode.READ: "https://r.jina.ai/",
    JinaReaderMode.SEARCH: "https://s.jina.ai/",
    JinaReaderMode.GROUND: "https://g.jina.ai/",
}


@component
class JinaReaderConnector:
    """
    A component that interacts with Jina AI's reader service to process queries and return documents.

    This component supports different modes of operation: `read`, `search`, and `ground`.

    Usage example:
    ```python
    from haystack_integrations.components.connectors.jina import JinaReaderConnector

    reader = JinaReaderConnector(mode="read")
    query = "https://example.com"
    result = reader.run(query=query)
    document = result["documents"][0]
    print(document.content)

    >>> "This domain is for use in illustrative examples..."
    ```
    """

    def __init__(
        self,
        mode: Union[JinaReaderMode, str],
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),  # noqa: B008
        json_response: bool = True,
    ):
        """
        Initialize a JinaReader instance.

        :param mode: The operation mode for the reader (`read`, `search` or `ground`).
            - `read`: process a URL and return the textual content of the page.
            - `search`: search the web and return textual content of the most relevant pages.
            - `ground`: call the grounding engine to perform fact checking.
            For more information on the modes, see the [Jina Reader documentation](https://jina.ai/reader/).
        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        :param json_response: Controls the response format from the Jina Reader API.
            If `True`, requests a JSON response, resulting in Documents with rich structured metadata.
            If `False`, requests a raw response, resulting in one Document with minimal metadata.
        """
        self.api_key = api_key
        self.json_response = json_response

        if isinstance(mode, str):
            mode = JinaReaderMode.from_str(mode)
        self.mode = mode

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            mode=str(self.mode),
            json_response=self.json_response,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JinaReaderConnector":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _json_to_document(self, data: dict) -> Document:
        """
        Convert a JSON response/record to a Document, depending on the reader mode.
        """
        if self.mode == JinaReaderMode.GROUND:
            content = data.pop("reason")
        else:
            content = data.pop("content")
        document = Document(content=content, meta=data)
        return document

    @component.output_types(documents=List[Document])
    def run(self, query: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, List[Document]]:
        """
        Process the query/URL using the Jina AI reader service.

        :param query: The query string or URL to process.
        :param headers: Optional headers to include in the request for customization. Refer to the
            [Jina Reader documentation](https://jina.ai/reader/) for more information.

        :returns:
            A dictionary with the following keys:
                - `documents`: A list of `Document` objects.
        """
        headers = headers or {}
        headers["Authorization"] = f"Bearer {self.api_key.resolve_value()}"

        if self.json_response:
            headers["Accept"] = "application/json"

        endpoint_url = READER_ENDPOINT_URL_BY_MODE[self.mode]
        encoded_target = quote(query, safe="")
        url = f"{endpoint_url}{encoded_target}"

        response = requests.get(url, headers=headers, timeout=60)

        # raw response: we just return a single Document with text
        if not self.json_response:
            meta = {"content_type": response.headers["Content-Type"], "query": query}
            return {"documents": [Document(content=response.text, meta=meta)]}

        response_json = json.loads(response.content).get("data", {})
        if self.mode == JinaReaderMode.SEARCH:
            documents = [self._json_to_document(record) for record in response_json]
            return {"documents": documents}

        return {"documents": [self._json_to_document(response_json)]}
