##### Haystack Implementation ######
# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import urllib
from typing import Any, Dict, Union

import requests
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from reader_mode import JinaReaderMode


@component
class JinaReader:
    """
    A component that interacts with Jina AI's reader service to process queries and return documents.

    This component supports different modes of operation: READ, SEARCH, and GROUND.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.readers.jina import JinaReader

    reader = JinaReader(mode="READ")
    query = "https://example.com"
    result = reader.run(query=query)
    document = result["document"]
    print(document.content)
    ```
    """
    def __init__(
        self,
        mode: Union[JinaReaderMode, str],
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    ):
        """
        Initialize a JinaReader instance.

        :param mode: The operation mode for the reader (READ, SEARCH, or GROUND).
        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        """
        resolved_api_key = api_key.resolve_value()
        self.api_key = api_key
        self.mode = mode

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            mode=self.mode,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JinaReader":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(document=Document)
    def run(self, query: str):
        """
        Process the query using the Jina AI reader service.

        :param query: The query string or URL to process.
        :returns: A list containing a single Document object with the processed content and metadata.
        """
        mode_map = {"READ": "r", "SEARCH": "s", "GROUND": "g"}
        mode = mode_map[self.mode]
        base_url = f"https://{mode}.jina.ai/"
        encoded_target = urllib.parse.quote(query, safe="")
        url = f"{base_url}{encoded_target}"
        response = self._session.get(url)
        metadata = {"content_type": response.headers["Content-Type"], "query": query}
        document = [Document(content=response.content, meta=metadata)]
        return document
