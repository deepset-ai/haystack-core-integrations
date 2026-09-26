# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import glob
import random
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from haystack import Document
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice
from PIL import Image

from haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_doc_image_embedder import (
    SentenceTransformersDocumentImageEmbedder,
)

IMPORT_PATH = (
    "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_doc_image_embedder"
)


class TestSentenceTransformersDocumentImageEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.trust_remote_code is False
        assert embedder.local_files_only is False
        assert embedder.precision == "float32"
        assert embedder._embedding_backend is None

    def test_init_with_parameters(self):
        embedder = SentenceTransformersDocumentImageEmbedder(
            file_path_meta_field="custom_file_path",
            root_path="root_path",
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            batch_size=64,
            progress_bar=False,
            trust_remote_code=True,
            local_files_only=True,
            precision="int8",
        )
        assert embedder.file_path_meta_field == "custom_file_path"
        assert embedder.root_path == "root_path"
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.trust_remote_code
        assert embedder.local_files_only
        assert embedder.precision == "int8"
        assert embedder.backend == "torch"
        assert embedder.model_kwargs is None
        assert embedder.tokenizer_kwargs is None
        assert embedder.config_kwargs is None
        assert embedder.encode_kwargs is None
        assert embedder._embedding_backend is None

    def test_to_dict(self):
        component = SentenceTransformersDocumentImageEmbedder(
            model="model", device=ComponentDevice.from_str("cpu"), model_kwargs={"torch_dtype": "torch.float32"}
        )
        data = component.to_dict()
        assert data == {
            "type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder",
            "init_parameters": {
                "file_path_meta_field": "file_path",
                "root_path": "",
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "local_files_only": False,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": None,
                "encode_kwargs": None,
                "config_kwargs": None,
                "precision": "float32",
                "backend": "torch",
            },
        }

    def test_from_dict(self):
        init_parameters = {
            "file_path_meta_field": "custom_file_path",
            "root_path": "root_path",
            "model": "model",
            "device": ComponentDevice.from_str("cuda:0").to_dict(),
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
            "batch_size": 64,
            "progress_bar": False,
            "normalize_embeddings": True,
            "trust_remote_code": True,
            "local_files_only": True,
            "model_kwargs": {"torch_dtype": "torch.float32"},
            "tokenizer_kwargs": {"model_max_length": 512},
            "config_kwargs": {"use_memory_efficient_attention": True},
            "precision": "int8",
        }
        component = SentenceTransformersDocumentImageEmbedder.from_dict(
            {"type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder", "init_parameters": init_parameters}
        )
        assert component.file_path_meta_field == "custom_file_path"
        assert component.root_path == "root_path"
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cuda:0")
        assert component.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.normalize_embeddings is True
        assert component.trust_remote_code
        assert component.local_files_only
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": True}
        assert component.precision == "int8"

    def test_from_dict_none_device(self):
        init_parameters = {
            "file_path_meta_field": "custom_file_path",
            "root_path": "root_path",
            "model": "model",
            "device": None,
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
            "batch_size": 64,
            "progress_bar": False,
            "normalize_embeddings": False,
            "trust_remote_code": True,
            "local_files_only": False,
            "precision": "float32",
        }
        component = SentenceTransformersDocumentImageEmbedder.from_dict(
            {"type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder", "init_parameters": init_parameters}
        )
        assert component.file_path_meta_field == "custom_file_path"
        assert component.root_path == "root_path"
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.trust_remote_code
        assert component.local_files_only is False
        assert component.precision == "float32"

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
        )
        mocked_factory.get_embedding_backend.assert_not_called()

        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs=None,
            processor_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
            backend="torch",
        )

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self, test_files_path):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        embedder._embedding_backend = MagicMock()
        embedder._embedding_backend.embed = lambda data, **_: [
            [random.random() for _ in range(16)] for _ in range(len(data))
        ]

        image_paths = glob.glob(str(test_files_path / "images" / "*.*")) + glob.glob(
            str(test_files_path / "pdf" / "*.pdf")
        )
        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"], strict=True):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert isinstance(new_doc.embedding[0], float)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentImageEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentImageEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=list_integers_input)

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersDocumentImageEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "onnx/model.onnx"},
            # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file
            # but this is to prevent a HF warning
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs={"file_name": "onnx/model.onnx"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersDocumentImageEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file
            # but this is to prevent a HF warning
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            processor_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(f"{IMPORT_PATH}._extract_image_sources_info")
    @patch(f"{IMPORT_PATH}._batch_convert_pdf_pages_to_images")
    @patch("PIL.Image.open")
    def test_run_none_images(
        self, mocked_pil_open, mocked_batch_convert_pdf_pages_to_images, mocked_extract_image_sources_info
    ):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        embedder._embedding_backend = MagicMock()

        mocked_extract_image_sources_info.return_value = [
            {"path": "doc1.pdf", "mime_type": "application/pdf", "page_number": 999},  # Page 999 doesn't exist
            {"path": "image1.jpg", "mime_type": "image/jpeg"},
        ]
        mocked_batch_convert_pdf_pages_to_images.return_value = {}  # Empty dict because page was skipped
        mocked_pil_open.return_value = Image.new("RGB", (100, 100))

        documents = [
            Document(content="PDF 1", meta={"file_path": "doc1.pdf", "page_number": 999}),
            Document(content="Image 1", meta={"file_path": "image1.jpg"}),
        ]

        with pytest.raises(RuntimeError, match=r"Conversion failed for some documents\."):
            embedder.run(documents=documents)

    @pytest.mark.integration
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason=(
            "This model does not play well with GitHub macOS runners and"
            "we prefer to avoid altering PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        ),
    )
    def test_live_run(self, test_files_path, del_hf_env_vars_if_empty):
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="sentence-transformers/clip-ViT-B-32", device=ComponentDevice.from_str("cpu")
        )

        documents = [
            Document(
                content="PDF document",
                meta={"file_path": str(test_files_path / "pdf" / "sample_pdf_1.pdf"), "page_number": 1},
            ),
            Document(content="Image document", meta={"file_path": str(test_files_path / "images" / "apple.jpg")}),
        ]

        result = embedder.run(documents=documents)
        assert len(result["documents"]) == len(documents)
        for doc, new_doc in zip(documents, result["documents"], strict=True):
            assert doc.embedding is None
            assert new_doc is not doc
            assert isinstance(new_doc, Document)
            assert isinstance(new_doc.embedding, list)
            assert len(new_doc.embedding) == 512
            assert all(isinstance(x, float) for x in new_doc.embedding)
            assert "embedding_source" not in doc.meta
            assert "embedding_source" in new_doc.meta
            assert new_doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in new_doc.meta["embedding_source"]


_PROCESSOR_UTIL_LOGGER = "haystack_integrations.utils.sentence_transformers"
_PROCESSOR_DEPRECATED_MESSAGE = "`tokenizer_kwargs` is deprecated. Use `processor_kwargs` instead."
_PROCESSOR_CONFLICT_SUFFIX = " Both were provided; `processor_kwargs` takes precedence."
_PROCESSOR_TYPE_NAME = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_doc_image_embedder.SentenceTransformersDocumentImageEmbedder"
_PROCESSOR_PATCH_TARGET = "haystack_integrations.components.embedders.sentence_transformers.sentence_transformers_doc_image_embedder._SentenceTransformersEmbeddingBackendFactory"


def _processor_util_warnings(caplog):
    return [
        record for record in caplog.records if record.name == _PROCESSOR_UTIL_LOGGER and record.levelname == "WARNING"
    ]


class TestSentenceTransformersDocumentImageEmbedderProcessorKwargs:
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
            component = SentenceTransformersDocumentImageEmbedder(
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
        component = SentenceTransformersDocumentImageEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        assert data["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
        assert "processor_kwargs" not in data["init_parameters"]
        legacy = SentenceTransformersDocumentImageEmbedder(model="model", tokenizer_kwargs={"model_max_length": 128})
        assert set(data["init_parameters"]) == set(legacy.to_dict()["init_parameters"])

    def test_from_dict_accepts_canonical_key_quietly(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "processor_kwargs": {"model_max_length": 128}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDocumentImageEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {"model_max_length": 128}
        assert _processor_util_warnings(caplog) == []

    def test_from_dict_legacy_only_is_quiet(self, caplog):
        data = {
            "type": _PROCESSOR_TYPE_NAME,
            "init_parameters": {"model": "model", "tokenizer_kwargs": {"model_max_length": 512}},
        }
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            component = SentenceTransformersDocumentImageEmbedder.from_dict(data)
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
            component = SentenceTransformersDocumentImageEmbedder.from_dict(data)
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
            component = SentenceTransformersDocumentImageEmbedder.from_dict(data)
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
            component = SentenceTransformersDocumentImageEmbedder.from_dict(data)
        assert component.tokenizer_kwargs == {}
        assert len(_processor_util_warnings(caplog)) == 1

    def test_canonical_round_trip_is_quiet(self, caplog):
        component = SentenceTransformersDocumentImageEmbedder(model="model", processor_kwargs={"model_max_length": 128})
        data = component.to_dict()
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            restored = SentenceTransformersDocumentImageEmbedder.from_dict(data)
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
    def test_warmup_forwards_effective_as_canonical(self, mocked_factory, legacy, canonical, expected):
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs=None,
            processor_kwargs=expected,
            config_kwargs=None,
            backend="torch",
        )
        assert "tokenizer_kwargs" not in mocked_factory.get_embedding_backend.call_args.kwargs

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_repeated_does_not_reconstruct_or_warn(self, mocked_factory, caplog):
        with caplog.at_level("WARNING", logger=_PROCESSOR_UTIL_LOGGER):
            embedder = SentenceTransformersDocumentImageEmbedder(
                model="model", tokenizer_kwargs={"model_max_length": 512}
            )
            embedder.warm_up()
            embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()
        assert len(_processor_util_warnings(caplog)) == 1

    @pytest.mark.parametrize(
        ("legacy", "canonical", "expected_length"),
        [
            pytest.param({"model_max_length": 128}, None, 128, id="legacy-length"),
            pytest.param(None, {"model_max_length": 128}, 128, id="canonical-length"),
            pytest.param({"model_max_length": 512}, {"model_max_length": 128}, 128, id="canonical-override"),
            pytest.param(None, None, None, id="both-none"),
            pytest.param({}, None, None, id="legacy-empty"),
            pytest.param({"model_max_length": 0}, None, None, id="legacy-zero"),
            pytest.param({"other": "value"}, None, None, id="legacy-without-length"),
            pytest.param({"model_max_length": 512}, {}, None, id="canonical-empty-suppresses"),
        ],
    )
    @patch(_PROCESSOR_PATCH_TARGET)
    def test_warmup_model_max_length_sync(self, mocked_factory, legacy, canonical, expected_length):
        mocked_factory.get_embedding_backend.return_value.model.max_seq_length = 999
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs=legacy,
            processor_kwargs=canonical,
        )
        embedder.warm_up()
        if expected_length is None:
            assert mocked_factory.get_embedding_backend.return_value.model.max_seq_length == 999
        else:
            assert mocked_factory.get_embedding_backend.return_value.model.max_seq_length == expected_length

    @patch(_PROCESSOR_PATCH_TARGET)
    def test_legacy_reassignment_before_warmup_is_effective(self, mocked_factory):
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="model", token=None, device=ComponentDevice.from_str("cpu")
        )
        embedder.tokenizer_kwargs = {"model_max_length": 128}
        embedder.warm_up()
        assert mocked_factory.get_embedding_backend.call_args.kwargs["processor_kwargs"] == {"model_max_length": 128}
        assert embedder.to_dict()["init_parameters"]["tokenizer_kwargs"] == {"model_max_length": 128}
