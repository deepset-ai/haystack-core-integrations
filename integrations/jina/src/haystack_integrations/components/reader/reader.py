
##### Haystack Implementation ######
# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

import requests
import urllib
from enum import Enum
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


class JinaReaderMode(Enum):
    """
    Enum representing modes for the Jina Reader.

    Modes:
        READ: For reading documents.
        SEARCH: For searching within documents.
        GROUND: For grounding or fact checking.
    """

    READ = "READ"
    SEARCH = "SEARCH"
    GROUND = "GROUND"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "JinaReaderMode":
        """
        Create the reader mode from a string.

        :param string:
            String to convert.
        :returns:
            Reader mode.
        """
        enum_map = {e.value: e for e in JinaReaderMode}
        reader_mode = enum_map.get(string)
        if reader_mode is None:
            msg = f"Unknown reader mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return reader_mode

@component
class JinaReader():

    def __init__(
        self,
        mode: Union[JinaReaderMode, str],
        url: Optional[str] = None,
        reader_query: Optional[str] = None,
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
    ):

        resolved_api_key = api_key.resolve_value()
        self.api_key = api_key
        self.mode = mode
        self.url = url
        self.reader_query = reader_query

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    @component.output_types(document=Document)
    def run(self, input:str):

        # check input depending on mode
        mode_map = {
            JinaReaderMode.READ: "r",
            JinaReaderMode.SEARCH: "s",
            JinaReaderMode.GROUND: "g"
        }
        base_url = "https://{}.jina.ai/".format(mode_map[self.mode])
        encoded_target = urllib.parse.quote(input, safe="")
        url = f"{base_url}{encoded_target}"
        response = self._session.get(
            url
        )
        metadata = {
            'content_type': response.headers['Content-Type'],
            'url':input
            }
        document = [Document(content = response.content, meta = metadata)]
        return document

        ... # do the rest and clean ups
        