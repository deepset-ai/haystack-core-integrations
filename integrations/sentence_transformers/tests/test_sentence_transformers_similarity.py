# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from haystack import Document
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice

from haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity import (
    SentenceTransformersSimilarityRanker,
)


class TestSentenceTransformersSimilarityRanker:
    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError):
            SentenceTransformersSimilarityRanker(top_k=-1)

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.CrossEncoder"
    )
    def test_init_warm_up_torch_backend(self, mocked_cross_encoder):
        ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="torch",
            trust_remote_code=True,
        )

        ranker.warm_up()
        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=True,
            model_kwargs=None,
            processor_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.CrossEncoder"
    )
    def test_init_warm_up_onnx_backend(self, mocked_cross_encoder):
        onnx_ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="onnx",
        )
        onnx_ranker.warm_up()

        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=False,
            model_kwargs=None,
            processor_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.CrossEncoder"
    )
    def test_init_warm_up_openvino_backend(self, mocked_cross_encoder):
        openvino_ranker = SentenceTransformersSimilarityRanker(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            backend="openvino",
        )
        openvino_ranker.warm_up()

        mocked_cross_encoder.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            token=None,
            trust_remote_code=False,
            model_kwargs=None,
            processor_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    def test_to_dict(self):
        component = SentenceTransformersSimilarityRanker()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "top_k": 10,
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "query_prefix": "",
                "query_suffix": "",
                "document_prefix": "",
                "document_suffix": "",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": True,
                "score_threshold": None,
                "trust_remote_code": False,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 16,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersSimilarityRanker(
            model="my_model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            top_k=5,
            query_prefix="query_instruction: ",
            query_suffix="\n",
            document_prefix="document_instruction: ",
            document_suffix="\n",
            scale_score=False,
            score_threshold=0.01,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer_kwargs={"model_max_length": 512},
            batch_size=32,
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": {"type": "single", "device": "cuda:0"},
                "model": "my_model",
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "top_k": 5,
                "query_prefix": "query_instruction: ",
                "query_suffix": "\n",
                "document_prefix": "document_instruction: ",
                "document_suffix": "\n",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": False,
                "score_threshold": 0.01,
                "trust_remote_code": True,
                "model_kwargs": {"torch_dtype": "torch.float16"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 32,
            },
        }

    def test_to_dict_with_quantization_options(self):
        component = SentenceTransformersSimilarityRanker(
            model_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "top_k": 10,
                "query_prefix": "",
                "query_suffix": "",
                "document_prefix": "",
                "document_suffix": "",
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": True,
                "score_threshold": None,
                "trust_remote_code": False,
                "model_kwargs": {
                    "load_in_4bit": True,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "torch.bfloat16",
                },
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 16,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {
                "device": ComponentDevice.resolve_device(None).to_dict(),
                "model": "my_model",
                "token": None,
                "top_k": 5,
                "query_prefix": "",
                "document_prefix": "",
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
                "scale_score": False,
                "score_threshold": 0.01,
                "trust_remote_code": False,
                "model_kwargs": {"torch_dtype": "torch.float16"},
                "tokenizer_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
                "batch_size": 32,
            },
        }

        component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "my_model"
        assert component.token is None
        assert component.top_k == 5
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert not component.scale_score
        assert component.score_threshold == 0.01
        assert component.trust_remote_code is False
        assert component.model_kwargs == {"torch_dtype": torch.float16}
        assert component.tokenizer_kwargs is None
        assert component.config_kwargs is None
        assert component.batch_size == 32

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker",
            "init_parameters": {},
        }

        component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.top_k == 10
        assert component.query_prefix == ""
        assert component.document_prefix == ""
        assert component.meta_fields_to_embed == []
        assert component.embedding_separator == "\n"
        assert component.scale_score
        assert component.score_threshold is None
        assert component.trust_remote_code is False
        assert component.model_kwargs is None
        assert component.tokenizer_kwargs is None
        assert component.config_kwargs is None
        assert component.backend == "torch"
        assert component.batch_size == 16

    @patch(
        "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.CrossEncoder"
    )
    def test_warm_up(self, mock_cross_encoder):
        ranker = SentenceTransformersSimilarityRanker()

        mock_cross_encoder.assert_not_called()

        ranker.warm_up()
        mock_cross_encoder.assert_called()

    def test_run_invalid_top_k(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()

        with pytest.raises(ValueError):
            ranker.run(query="test", documents=[Document(content="document")], top_k=-1)

    def test_run_top_k_zero_raises(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()

        with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
            ranker.run(query="test", documents=[Document(content="document")], top_k=0)

    def test_run_scale_score_false_overrides_init(self):
        ranker = SentenceTransformersSimilarityRanker()  # scale_score=True by default
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.rank.return_value = [{"corpus_id": 0, "score": 1.0}]
        ranker._cross_encoder = mock_cross_encoder

        ranker.run(query="test", documents=[Document(content="document")], scale_score=False)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert isinstance(kwargs["activation_fn"], torch.nn.Identity)

    def test_run_score_threshold_zero_overrides_init(self):
        ranker = SentenceTransformersSimilarityRanker(score_threshold=0.5)
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.rank.return_value = [{"corpus_id": 0, "score": 0.2}, {"corpus_id": 1, "score": -0.4}]
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content="first"), Document(content="second")]
        result = ranker.run(query="test", documents=documents, score_threshold=0.0)

        assert [doc.score for doc in result["documents"]] == [0.2]

    def test_returns_empty_list_if_no_documents_are_provided(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()

        output = ranker.run(query="City in Germany", documents=[])
        assert not output["documents"]

    def test_embed_meta(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model", meta_fields_to_embed=["meta_field"], embedding_separator="\n"
        )
        mock_cross_encoder = MagicMock()
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["query"] == "test"
        assert kwargs["documents"] == [
            "meta_value 0\ndocument number 0",
            "meta_value 1\ndocument number 1",
            "meta_value 2\ndocument number 2",
            "meta_value 3\ndocument number 3",
            "meta_value 4\ndocument number 4",
        ]
        assert kwargs["batch_size"] == 16
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["return_documents"] is False

    def test_embed_meta_preserves_falsy_values(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model",
            meta_fields_to_embed=["rating", "is_available", "missing", "none_value"],
            embedding_separator="\n",
        )
        mock_cross_encoder = MagicMock()
        mock_cross_encoder.rank.return_value = [{"corpus_id": 0, "score": 1.0}]
        ranker._cross_encoder = mock_cross_encoder
        documents = [Document(content="document", meta={"rating": 0, "is_available": False, "none_value": None})]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["documents"] == ["0\nFalse\ndocument"]

    def test_prefix(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model", query_prefix="query_instruction: ", document_prefix="document_instruction: "
        )
        mock_cross_encoder = MagicMock()
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["query"] == "query_instruction: test"
        assert kwargs["documents"] == [
            "document_instruction: document number 0",
            "document_instruction: document number 1",
            "document_instruction: document number 2",
            "document_instruction: document number 3",
            "document_instruction: document number 4",
        ]
        assert kwargs["batch_size"] == 16
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["convert_to_numpy"] is True

    def test_suffix(self):
        ranker = SentenceTransformersSimilarityRanker(
            model="model", query_suffix="\nquery_suffix", document_suffix="\ndocument_suffix"
        )
        mock_cross_encoder = MagicMock()
        ranker._cross_encoder = mock_cross_encoder

        documents = [Document(content=f"document number {i}", meta={"meta_field": f"meta_value {i}"}) for i in range(5)]

        ranker.run(query="test", documents=documents)

        _, kwargs = mock_cross_encoder.rank.call_args
        assert kwargs["query"] == "test\nquery_suffix"
        assert kwargs["documents"] == [
            "document number 0\ndocument_suffix",
            "document number 1\ndocument_suffix",
            "document number 2\ndocument_suffix",
            "document number 3\ndocument_suffix",
            "document number 4\ndocument_suffix",
        ]
        assert kwargs["batch_size"] == 16
        assert isinstance(kwargs["activation_fn"], torch.nn.Sigmoid)
        assert kwargs["convert_to_numpy"] is True

    def test_scale_score_false(self):
        mock_cross_encoder = MagicMock()
        ranker = SentenceTransformersSimilarityRanker(model="model", scale_score=False)
        ranker._cross_encoder = mock_cross_encoder

        mock_cross_encoder.rank.return_value = [{"score": -10.6859, "corpus_id": 0}, {"score": -8.9874, "corpus_id": 1}]

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert out["documents"][0].score == pytest.approx(-10.6859, abs=1e-4)
        assert out["documents"][1].score == pytest.approx(-8.9874, abs=1e-4)

    def test_score_threshold(self):
        mock_cross_encoder = MagicMock()
        ranker = SentenceTransformersSimilarityRanker(model="model", scale_score=False, score_threshold=0.1)
        ranker._cross_encoder = mock_cross_encoder

        mock_cross_encoder.rank.return_value = [{"score": 0.955, "corpus_id": 0}, {"score": 0.001, "corpus_id": 1}]

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert len(out["documents"]) == 1

        documents = [Document(content="document number 0"), Document(content="document number 1")]
        out = ranker.run(query="test", documents=documents)
        assert len(out["documents"]) == 1

    def test_scores_cast_to_python_float_when_numpy_scalars_returned(self):
        mock_cross_encoder = MagicMock()
        ranker = SentenceTransformersSimilarityRanker(model="model")
        ranker._cross_encoder = mock_cross_encoder

        # Simulate backend returning numpy scalar types
        mock_cross_encoder.rank.return_value = [
            {"score": np.float32(0.123), "corpus_id": 0},
            {"score": np.float64(0.456), "corpus_id": 1},
        ]

        documents = [Document(content="doc 0"), Document(content="doc 1")]
        out = ranker.run(query="test", documents=documents)

        assert len(out["documents"]) == 2
        for d in out["documents"]:
            assert isinstance(d.score, float)

    def test_run_deduplicates_documents(self):
        ranker = SentenceTransformersSimilarityRanker()
        ranker._cross_encoder = MagicMock()
        ranker._cross_encoder.rank.return_value = [{"score": 0.42, "corpus_id": 0}, {"score": 0.12, "corpus_id": 1}]

        documents = [
            Document(id="duplicate", content="keep me", score=0.9),
            Document(id="duplicate", content="drop me", score=0.1),
            Document(id="unique", content="unique"),
        ]
        result = ranker.run(query="test", documents=documents)
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "keep me"
        assert result["documents"][1].content == "unique"

    @pytest.mark.integration
    def test_run(self, del_hf_env_vars_if_empty):
        ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device=ComponentDevice.from_str("cpu")
        )

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"
        expected_scores = [0.14568544924259186, 0.18189962208271027, 0.5728498697280884]

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 3
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted(expected_scores, reverse=True)
        assert docs_after[0].score == pytest.approx(sorted_scores[0], abs=1e-6)
        assert docs_after[1].score == pytest.approx(sorted_scores[1], abs=1e-6)
        assert docs_after[2].score == pytest.approx(sorted_scores[2], abs=1e-6)

        for doc in docs_after:
            assert isinstance(doc.score, float)

    @pytest.mark.integration
    def test_run_top_k(self, del_hf_env_vars_if_empty):
        ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce", top_k=2, device=ComponentDevice.from_str("cpu")
        )

        query = "City in Bosnia and Herzegovina"
        docs_before_texts = ["Berlin", "Belgrade", "Sarajevo"]
        expected_first_text = "Sarajevo"

        docs_before = [Document(content=text) for text in docs_before_texts]
        output = ranker.run(query=query, documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 2
        assert docs_after[0].content == expected_first_text

        sorted_scores = sorted([doc.score for doc in docs_after], reverse=True)
        assert [doc.score for doc in docs_after] == sorted_scores

        for doc in docs_after:
            assert isinstance(doc.score, float)

    @pytest.mark.integration
    def test_run_single_document(self, del_hf_env_vars_if_empty):
        ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce", device=ComponentDevice.from_str("cpu")
        )
        docs_before = [Document(content="Berlin")]
        output = ranker.run(query="City in Germany", documents=docs_before)
        docs_after = output["documents"]

        assert len(docs_after) == 1
        assert isinstance(docs_after[0].score, float)


_PROCESSOR_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_PROCESSOR_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_PROCESSOR_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."
_PROCESSOR_TYPE_NAME = "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.SentenceTransformersSimilarityRanker"
_PROCESSOR_PATCH_TARGET = (
    "haystack_integrations.components.rankers.sentence_transformers.sentence_transformers_similarity.CrossEncoder"
)


def _processor_util_warnings(caplog):
    return [
        record for record in caplog.records if record.name == _PROCESSOR_UTIL_LOGGER and record.levelname == "WARNING"
    ]


class TestSentenceTransformersSimilarityRankerProcessorKwargs:
    @pytest.mark.parametrize(
        ("legacy", "canonical", "expect_warn", "expect_conflict"),
        [
            pytest.param(None, None, False, False, id="both-none"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, False, False, id="canonical-only"),
            pytest.param({}, None, True, False, id="legacy-empty"),
            pytest.param(None, {}, False, False, id="canonical-empty"),
            pytest.param({"model_max_length": 512}, {}, True, True, id="legacy-plus-canonical-empty"),
            pytest.param({"model_max_length": 512}, {"model_max_length": 128}, True, True, id="both-nonempty"),
            pytest.param({"model_max_length": 512}, None, True, False, id="legacy-plus-explicit-canonical-none"),
        ],
    )
    def test_constructor_matrix(self, caplog, legacy, canonical, expect_warn, expect_conflict):
        legacy_snapshot = None if legacy is None else dict(legacy)
        canonical_snapshot = None if canonical is None else dict(canonical)
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker(
                model="model",
                token=None,
                device=ComponentDevice.from_str("cpu"),
                tokenizer_kwargs=legacy,
                processor_kwargs=canonical,
            )
        expected = canonical if canonical is not None else legacy
        assert component.tokenizer_kwargs == expected
        if expected is None:
            assert component.tokenizer_kwargs is None
        elif canonical is not None:
            assert component.tokenizer_kwargs is canonical
        else:
            assert component.tokenizer_kwargs is legacy
        assert not hasattr(component, "processor_kwargs")
        assert legacy == legacy_snapshot
        assert canonical == canonical_snapshot
        records = _processor_util_warnings(caplog)
        if expect_warn:
            assert len(records) == 1
            expected_message = _PROCESSOR_DEPRECATED_MESSAGE
            if expect_conflict:
                expected_message += _PROCESSOR_CONFLICT_SUFFIX
            assert records[0].getMessage() == expected_message
        else:
            assert records == []

    def test_to_dict_emits_legacy_key_only(self):
        component = SentenceTransformersSimilarityRanker(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in data["init_parameters"]
        legacy = SentenceTransformersSimilarityRanker(model="model", tokenizer_kwargs={"model_max_length": 128})
        assert set(data["init_parameters"]) == set(legacy.to_dict()["init_parameters"])

    def test_from_dict_accepts_canonical_key_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "processor_kwargs": {"model_max_length": 128}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_legacy_only_is_quiet(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "tokenizer_kwargs": {"model_max_length": 512}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_both_keys_canonical_wins_with_warning(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {"model_max_length": 128},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        records = _processor_util_warnings(caplog)
        assert len(records) == 1
        assert records[0].getMessage() == _PROCESSOR_DEPRECATED_MESSAGE + _PROCESSOR_CONFLICT_SUFFIX
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 512}
        assert data["init_parameters"]["processor_kwargs"] == {"model_max_length": 128}
        reserialized = component.to_dict()
        assert reserialized["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in reserialized["init_parameters"]

    def test_from_dict_explicit_canonical_none_falls_back_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": None,
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_canonical_empty_overrides_legacy(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {
                "model": "model",
                "tokenizer_kwargs": {"model_max_length": 512},
                "processor_kwargs": {},
            },
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersSimilarityRanker.from_dict(data)
        assert component.tokenizer_kwargs == {}
        assert len(_processor_util_warnings(caplog)) == 1

    def test_canonical_round_trip_is_quiet(self, caplog):
        component = SentenceTransformersSimilarityRanker(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            restored = SentenceTransformersSimilarityRanker.from_dict(data)
        assert restored.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    @pytest.mark.parametrize(
        ("legacy", "canonical", "expected"),
        [
            pytest.param(None, None, None, id="both-none"),
            pytest.param({"model_max_length": 512}, None, {"model_max_length": 512}, id="legacy-only"),
            pytest.param(None, {"model_max_length": 128}, {"model_max_length": 128}, id="canonical-only"),
            pytest.param(
                {"model_max_length": 512}, {"model_max_length": 128}, {"model_max_length": 128}, id="both-nonempty"
            ),
            pytest.param({"model_max_length": 512}, {}, {}, id="canonical-empty-wins"),
        ],
    )
    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_forwards_effective_as_canonical(self, mocked_sdk, legacy, canonical, expected):
        ranker = SentenceTransformersSimilarityRanker(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        ranker.warm_up()
        mocked_sdk.assert_called_once_with(
            model_name_or_path="model",
            device="cpu",
            token=None,
            trust_remote_code=False,
            model_kwargs=None,
            processor_kwargs=expected,
            config_kwargs=None,
            backend="torch",
        )
        assert "tokenizer_kwargs" not in mocked_sdk.call_args.kwargs

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_repeated_does_not_reconstruct_or_warn(self, mocked_sdk, caplog):
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            ranker = SentenceTransformersSimilarityRanker(model="model", tokenizer_kwargs={"model_max_length": 512})
            ranker.warm_up()
            ranker.warm_up()
        mocked_sdk.assert_called_once()
        assert len(_processor_util_warnings(caplog)) == 1

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_legacy_reassignment_before_warmup_is_effective(self, mocked_sdk):
        ranker = SentenceTransformersSimilarityRanker(model="model", token=None, device=ComponentDevice.from_str("cpu"))
        ranker.tokenizer_kwargs = {"model_max_length": 128}
        ranker.warm_up()
        assert mocked_sdk.call_args.kwargs["processor_kwargs"] == {"model_max_length": 128}

    @pytest.mark.integration
    @pytest.mark.parametrize("kwargs_name", ["tokenizer_kwargs", "processor_kwargs"])
    def test_processor_kwargs_integration(self, kwargs_name, del_hf_env_vars_if_empty):
        ranker = SentenceTransformersSimilarityRanker(
            model="cross-encoder-testing/reranker-bert-tiny-gooaq-bce",
            device=ComponentDevice.from_str("cpu"),
            **{kwargs_name: {"model_max_length": 128}},
        )
        ranker.warm_up()
        assert ranker._cross_encoder is not None
        assert ranker._cross_encoder.tokenizer.model_max_length == 128
        docs_before = [Document(content=text) for text in ["Berlin", "Belgrade", "Sarajevo"]]
        output = ranker.run(query="City in Bosnia and Herzegovina", documents=docs_before)
        docs_after = output["documents"]
        assert len(docs_after) == 3
        for doc in docs_after:
            assert isinstance(doc.score, float)
        assert [doc.score for doc in docs_after] == sorted([doc.score for doc in docs_after], reverse=True)
