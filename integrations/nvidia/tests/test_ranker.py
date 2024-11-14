# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from typing import Any, Optional, Union

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.rankers.nvidia import NvidiaRanker
from haystack_integrations.components.rankers.nvidia.ranker import _DEFAULT_MODEL
from haystack_integrations.components.rankers.nvidia.truncate import RankerTruncateMode


class TestNvidiaRanker:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        client = NvidiaRanker()
        assert client._model == _DEFAULT_MODEL
        assert client._api_key == Secret.from_env_var("NVIDIA_API_KEY")

    def test_init_with_parameters(self):
        client = NvidiaRanker(
            api_key=Secret.from_token("fake-api-key"),
            model=_DEFAULT_MODEL,
            top_k=3,
            truncate="END",
        )
        assert client._api_key == Secret.from_token("fake-api-key")
        assert client._model == _DEFAULT_MODEL
        assert client._top_k == 3
        assert client._truncate == RankerTruncateMode.END

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        client = NvidiaRanker()
        with pytest.raises(ValueError):
            client.warm_up()

    def test_init_pass_wo_api_key_w_api_url(self):
        url = "https://url.bogus/v1"
        client = NvidiaRanker(api_url=url)
        assert client._api_url == url

    def test_warm_up_required(self):
        client = NvidiaRanker()
        with pytest.raises(RuntimeError) as e:
            client.run("query", [Document(content="doc")])
        assert "not been loaded" in str(e.value)

    @pytest.mark.parametrize(
        "truncate",
        [
            None,
            "END",
            "NONE",
            RankerTruncateMode.END,
            RankerTruncateMode.NONE,
        ],
        ids=["None", "END-str", "NONE-str", "END-enum", "NONE-enum"],
    )
    def test_mocked(
        self,
        requests_mock,
        monkeypatch,
        truncate: Optional[Union[RankerTruncateMode, str]],
    ) -> None:
        query = "What is it?"
        documents = [
            Document(content="Nothing really."),
            Document(content="Maybe something."),
            Document(content="Not this."),
        ]

        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        requests_mock.post(
            re.compile(r".*ranking"),
            json={
                "rankings": [
                    {"index": 1, "logit": 4.2},
                    {"index": 0, "logit": 2.4},
                    {"index": 2, "logit": -4.2},
                ]
            },
        )

        truncate_param = {}
        if truncate:
            truncate_param = {"truncate": truncate}
        client = NvidiaRanker(
            top_k=2,
            **truncate_param,
        )
        client.warm_up()

        response = client.run(
            query=query,
            documents=documents,
        )["documents"]

        assert requests_mock.last_request is not None
        request_payload = requests_mock.last_request.json()
        if truncate is None:
            assert "truncate" not in request_payload
        else:
            assert "truncate" in request_payload
            assert request_payload["truncate"] == str(truncate)

        assert len(response) == 2
        assert response[0].content == documents[1].content
        assert response[0].score == 4.2
        assert response[1].content == documents[0].content
        assert response[1].score == 2.4

        response = client.run(
            query=query,
            documents=documents,
            top_k=1,
        )["documents"]
        assert len(response) == 1
        assert response[0].content == documents[1].content
        assert response[0].score == 4.2

    @pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "START", "BOGUS"])
    def test_truncate_invalid(self, truncate: Any) -> None:
        with pytest.raises(ValueError) as e:
            NvidiaRanker(truncate=truncate)
        assert "not a valid RankerTruncateMode" in str(e.value)

    @pytest.mark.parametrize("top_k", [1.0, "BOGUS"])
    def test_top_k_invalid(self, monkeypatch, top_k: Any) -> None:
        with pytest.raises(TypeError) as e:
            NvidiaRanker(top_k=top_k)
        assert "parameter to be an integer" in str(e.value)

        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        client = NvidiaRanker()
        client.warm_up()
        with pytest.raises(TypeError) as e:
            client.run("query", [Document(content="doc")], top_k=top_k)
        assert "parameter to be an integer" in str(e.value)

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration(
        self,
    ) -> None:
        query = "What is it?"
        documents = [
            Document(content="Nothing really."),
            Document(content="Maybe something."),
            Document(content="Not this."),
        ]

        client = NvidiaRanker(top_k=2)
        client.warm_up()

        response = client.run(query=query, documents=documents)["documents"]

        assert len(response) == 2
        assert {response[0].content, response[1].content} == {documents[0].content, documents[1].content}

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_NIM_RANKER_MODEL", None)
        or not os.environ.get("NVIDIA_NIM_RANKER_ENDPOINT_URL", None),
        reason="Export an env var called NVIDIA_NIM_RANKER_MODEL containing the hosted model name and "
        "NVIDIA_NIM_RANKER_ENDPOINT_URL containing the local URL to call.",
    )
    @pytest.mark.integration
    def test_nim_integration(self):
        query = "What is it?"
        documents = [
            Document(content="Nothing really."),
            Document(content="Maybe something."),
            Document(content="Not this."),
        ]

        client = NvidiaRanker(
            model=os.environ["NVIDIA_NIM_RANKER_MODEL"],
            api_url=os.environ["NVIDIA_NIM_RANKER_ENDPOINT_URL"],
            top_k=2,
        )
        client.warm_up()

        response = client.run(query=query, documents=documents)["documents"]

        assert len(response) == 2
        assert {response[0].content, response[1].content} == {documents[0].content, documents[1].content}

    def test_top_k_warn(self, monkeypatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        client = NvidiaRanker(top_k=0)
        client.warm_up()
        with pytest.warns(UserWarning) as record0:
            client.run("query", [Document(content="doc")])
        assert "top_k should be at least 1" in str(record0[0].message)

        client = NvidiaRanker(top_k=1)
        client.warm_up()
        with pytest.warns(UserWarning) as record1:
            client.run("query", [Document(content="doc")], top_k=0)
        assert "top_k should be at least 1" in str(record1[0].message)

    def test_model_typeerror(self) -> None:
        with pytest.raises(TypeError) as e:
            NvidiaRanker(model=1)
        assert "parameter to be a string" in str(e.value)

    def test_api_url_typeerror(self) -> None:
        with pytest.raises(TypeError) as e:
            NvidiaRanker(api_url=1)
        assert "parameter to be a string" in str(e.value)

    def test_query_typeerror(self, monkeypatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        client = NvidiaRanker()
        client.warm_up()
        with pytest.raises(TypeError) as e:
            client.run(1, [Document(content="doc")])
        assert "parameter to be a string" in str(e.value)

    def test_documents_typeerror(self, monkeypatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        client = NvidiaRanker()
        client.warm_up()
        with pytest.raises(TypeError) as e:
            client.run("query", "doc")
        assert "parameter to be a list" in str(e.value)

        with pytest.raises(TypeError) as e:
            client.run("query", [1])
        assert "list of Document objects" in str(e.value)

    def test_top_k_typeerror(self, monkeypatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        client = NvidiaRanker()
        client.warm_up()
        with pytest.raises(TypeError) as e:
            client.run("query", [Document(content="doc")], top_k="1")
        assert "parameter to be an integer" in str(e.value)

    def test_model_unknown(self) -> None:
        with pytest.raises(ValueError) as e:
            NvidiaRanker(model="unknown-model")
        assert "unknown" in str(e.value)

    def test_warm_up_once(self, monkeypatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        client = NvidiaRanker()
        client.warm_up()
        backend = client._backend
        client.warm_up()
        assert backend == client._backend

    def test_to_dict(self) -> None:
        client = NvidiaRanker()
        assert client.to_dict() == {
            "type": "haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker",
            "init_parameters": {
                "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                "top_k": 5,
                "truncate": None,
                "api_url": None,
                "api_key": {"type": "env_var", "env_vars": ["NVIDIA_API_KEY"], "strict": True},
            },
        }

    def test_from_dict(self) -> None:
        client = NvidiaRanker.from_dict(
            {
                "type": "haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker",
                "init_parameters": {
                    "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                    "top_k": 5,
                    "truncate": None,
                    "api_url": None,
                    "api_key": {"type": "env_var", "env_vars": ["NVIDIA_API_KEY"], "strict": True},
                },
            }
        )
        assert client._model == "nvidia/nv-rerankqa-mistral-4b-v3"
        assert client._top_k == 5
        assert client._truncate is None
        assert client._api_url is None
        assert client._api_key == Secret.from_env_var("NVIDIA_API_KEY")

    def test_from_dict_defaults(self) -> None:
        client = NvidiaRanker.from_dict(
            {
                "type": "haystack_integrations.components.rankers.nvidia.ranker.NvidiaRanker",
                "init_parameters": {},
            }
        )
        assert client._model == "nvidia/nv-rerankqa-mistral-4b-v3"
        assert client._top_k == 5
        assert client._truncate is None
        assert client._api_url is None
        assert client._api_key == Secret.from_env_var("NVIDIA_API_KEY")
