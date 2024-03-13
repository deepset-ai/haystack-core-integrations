import copy
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from haystack.utils import Secret

FUNCTIONS_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/functions"
INVOKE_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions"
STATUS_ENDPOINT = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status"

ACCEPTED_STATUS_CODE = 202


@dataclass
class AvailableNvidiaCloudFunctions:
    name: str
    id: str
    status: Optional[str] = None


class NvidiaCloudFunctionsClient:
    def __init__(self, *, api_key: Secret, headers: Dict[str, str], timeout: int = 60):
        self.api_key = api_key.resolve_value()
        if self.api_key is None:
            msg = "Nvidia Cloud Functions API key is not set."
            raise ValueError(msg)

        self.fetch_url_format = STATUS_ENDPOINT
        self.headers = copy.deepcopy(headers)
        self.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
            }
        )
        self.timeout = timeout
        self.session = requests.Session()

    def query_function(self, func_id: str, payload: Dict[str, str]) -> Dict[str, str]:
        invoke_url = f"{INVOKE_ENDPOINT}/{func_id}"

        response = self.session.post(invoke_url, headers=self.headers, json=payload, timeout=self.timeout)
        request_id = response.headers.get("NVCF-REQID")
        if request_id is None:
            msg = "NVCF-REQID header not found in response"
            raise ValueError(msg)

        while response.status_code == ACCEPTED_STATUS_CODE:
            fetch_url = f"{self.fetch_url_format}/{request_id}"
            response = self.session.get(fetch_url, headers=self.headers, timeout=self.timeout)

        response.raise_for_status()
        return response.json()

    def available_functions(self) -> Dict[str, AvailableNvidiaCloudFunctions]:
        response = self.session.get(FUNCTIONS_ENDPOINT, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        return {
            f["name"]: AvailableNvidiaCloudFunctions(
                name=f["name"],
                id=f["id"],
                status=f.get("status"),
            )
            for f in response.json()["functions"]
        }

    def get_model_nvcf_id(self, model: str) -> str:
        """
        Returns the Nvidia Cloud Functions UUID for the given model.
        """

        available_functions = self.available_functions()
        func = available_functions.get(model)
        if func is None:
            msg = f"Model '{model}' was not found on the Nvidia Cloud Functions backend"
            raise ValueError(msg)
        elif func.status != "ACTIVE":
            msg = f"Model '{model}' is not currently active/usable on the Nvidia Cloud Functions backend"
            raise ValueError(msg)

        return func.id
