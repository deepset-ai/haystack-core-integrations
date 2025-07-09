from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.embedders.instructor_embedders import InstructorTextEmbedder


class TestInstructorTextEmbedder:
    def test_init_default(self):
        """
        Test default initialization parameters for InstructorTextEmbedder.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-base")
        assert embedder.model == "hkunlp/instructor-base"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.instruction == "Represent the sentence"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    def test_init_with_parameters(self):
        """
        Test custom initialization parameters for InstructorTextEmbedder.
        """
        embedder = InstructorTextEmbedder(
            model="hkunlp/instructor-base",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            instruction="Represent the 'domain' 'text_type' for 'task_objective'",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        assert embedder.model == "hkunlp/instructor-base"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    def test_to_dict(self):
        """
        Test serialization of InstructorTextEmbedder to a dictionary, using default initialization parameters.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-base", device=ComponentDevice.from_str("cpu"))
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder.InstructorTextEmbedder",  # noqa
            "init_parameters": {
                "model": "hkunlp/instructor-base",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "instruction": "Represent the sentence",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        """
        Test serialization of InstructorTextEmbedder to a dictionary, using custom initialization parameters.
        """
        embedder = InstructorTextEmbedder(
            model="hkunlp/instructor-base",
            device=ComponentDevice.from_str("cuda:0"),
            instruction="Represent the financial document for retrieval",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
        )
        embedder_dict = embedder.to_dict()
        assert embedder_dict == {
            "type": "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder.InstructorTextEmbedder",  # noqa
            "init_parameters": {
                "model": "hkunlp/instructor-base",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }

    def test_from_dict(self):
        """
        Test deserialization of InstructorTextEmbedder from a dictionary, using default initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder.InstructorTextEmbedder",  # noqa
            "init_parameters": {
                "model": "hkunlp/instructor-base",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "instruction": "Represent the 'domain' 'text_type' for 'task_objective'",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
            },
        }
        embedder = InstructorTextEmbedder.from_dict(embedder_dict)
        assert embedder.model == "hkunlp/instructor-base"
        assert embedder.device == ComponentDevice.from_str("cpu")
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.instruction == "Represent the 'domain' 'text_type' for 'task_objective'"
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False

    def test_from_dict_with_custom_init_parameters(self):
        """
        Test deserialization of InstructorTextEmbedder from a dictionary, using custom initialization parameters.
        """
        embedder_dict = {
            "type": "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder.InstructorTextEmbedder",  # noqa
            "init_parameters": {
                "model": "hkunlp/instructor-base",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "instruction": "Represent the financial document for retrieval",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        }
        embedder = InstructorTextEmbedder.from_dict(embedder_dict)
        assert embedder.model == "hkunlp/instructor-base"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.instruction == "Represent the financial document for retrieval"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True

    @patch(
        "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder._InstructorEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        """
        Test for checking embedder instances after warm-up.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-base", device=ComponentDevice.from_str("cpu"))
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="hkunlp/instructor-base",
            device="cpu",
            token=Secret.from_env_var("HF_API_TOKEN", strict=False),
        )

    @patch(
        "haystack_integrations.components.embedders.instructor_embedders.instructor_text_embedder._InstructorEmbeddingBackendFactory"
    )
    def test_warmup_does_not_reload(self, mocked_factory):
        """
        Test for checking backend instances after multiple warm-ups.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-base")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_embed(self):
        """
        Test for checking output dimensions and embedding dimensions.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-large")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()  # noqa: ARG005

        text = "Good text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(emb, float) for emb in embedding)

    def test_run_wrong_incorrect_format(self):
        """
        Test for checking incorrect input format when creating embedding.
        """
        embedder = InstructorTextEmbedder(model="hkunlp/instructor-large")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="InstructorTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run(self):
        embedder = InstructorTextEmbedder(
            model="hkunlp/instructor-base",
            device=ComponentDevice.from_str("cpu"),
            instruction="Represent the Science sentence for retrieval",
        )
        embedder.warm_up()

        text = "Parton energy loss in QCD matter"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(emb, float) for emb in embedding)
