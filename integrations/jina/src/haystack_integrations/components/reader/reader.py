######## Jina Reader Docs Implementation ########
######## Read ########
import requests

url = 'https://r.jina.ai/https://example.com'
headers = {
    'Authorization': 'Bearer jina_123456'
}

response = requests.get(url, headers=headers)
print(response.text)

######### Search ##########
url = 'https://s.jina.ai/When%20was%20Jina%20AI%20founded?'
headers = {
    'Authorization': 'Bearer jina_123456'
}

response = requests.get(url, headers=headers)

print(response.text)

###### Ground #######
url = 'https://g.jina.ai/Jina%20AI%20was%20founded%20in%202020%20in%20Berlin.'
headers = {
    'Accept': 'application/json',
    'Authorization': 'Bearer jina_123456'
}

response = requests.get(url, headers=headers)

print(response.text)
##### End Jina Docs ######

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
    Specifies how inputs to the NVIDIA embedding components are truncated.
    If START, the input will be truncated from the start.
    If END, the input will be truncated from the end.
    If NONE, an error will be returned (if the input is too long).
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
        api_key: Secret = Secret.from_env_var("JINA_API_KEY"),
        mode: Union[JinaReaderMode, str],
        url: Optional[str],
        reader_query: Optional[str]
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
            url,
            headers=headers
        )

        ... # do the rest and clean ups
        