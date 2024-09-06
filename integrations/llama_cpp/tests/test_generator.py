import os
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator


@pytest.fixture
def model_path():
    return Path(__file__).parent / "models"


def download_file(file_link, filename, capsys):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)  # noqa: S310
        with capsys.disabled():
            print("\nModel file downloaded successfully.")
    else:
        with capsys.disabled():
            print("\nModel file already exists.")


class TestLlamaCppGenerator:
    @pytest.fixture
    def generator(self, model_path, capsys):
        ggml_model_path = (
            "https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210.Q3_K_S.gguf"
        )
        filename = "openchat-3.5-1210.Q3_K_S.gguf"

        # Download GGUF model from HuggingFace
        download_file(ggml_model_path, str(model_path / filename), capsys)

        model_path = str(model_path / filename)
        generator = LlamaCppGenerator(model=model_path, n_ctx=128, n_batch=128)
        generator.warm_up()
        return generator

    @pytest.fixture
    def generator_mock(self):
        mock_model = MagicMock()
        generator = LlamaCppGenerator(model="test_model.gguf", n_ctx=2048, n_batch=512)
        generator.model = mock_model
        return generator, mock_model

    def test_default_init(self):
        """
        Test default initialization parameters.
        """
        generator = LlamaCppGenerator(model="test_model.gguf")

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 0
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 0, "n_batch": 512}
        assert generator.generation_kwargs == {}

    def test_custom_init(self):
        """
        Test custom initialization parameters.
        """
        generator = LlamaCppGenerator(
            model="test_model.gguf",
            n_ctx=2048,
            n_batch=512,
        )

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 2048
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 2048, "n_batch": 512}
        assert generator.generation_kwargs == {}

    def test_ignores_model_path_if_specified_in_model_kwargs(self):
        """
        Test that model_path is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppGenerator(
            model="test_model.gguf",
            n_ctx=512,
            n_batch=512,
            model_kwargs={"model_path": "other_model.gguf"},
        )
        assert generator.model_kwargs["model_path"] == "other_model.gguf"

    def test_ignores_n_ctx_if_specified_in_model_kwargs(self):
        """
        Test that n_ctx is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppGenerator(model="test_model.gguf", n_ctx=512, n_batch=512, model_kwargs={"n_ctx": 1024})
        assert generator.model_kwargs["n_ctx"] == 1024

    def test_ignores_n_batch_if_specified_in_model_kwargs(self):
        """
        Test that n_batch is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppGenerator(model="test_model.gguf", n_ctx=512, n_batch=512, model_kwargs={"n_batch": 1024})
        assert generator.model_kwargs["n_batch"] == 1024

    def test_raises_error_without_warm_up(self):
        """
        Test that the generator raises an error if warm_up() is not called before running.
        """
        generator = LlamaCppGenerator(model="test_model.gguf", n_ctx=512, n_batch=512)
        with pytest.raises(RuntimeError):
            generator.run("What is the capital of China?")

    def test_run_with_empty_prompt(self, generator_mock):
        """
        Test that an empty prompt returns an empty list of replies.
        """
        generator, _ = generator_mock
        result = generator.run("")
        assert result["replies"] == []

    def test_run_with_valid_prompt(self, generator_mock):
        """
        Test that a valid prompt returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "choices": [{"text": "Generated text"}],
            "metadata": {"other_info": "Some metadata"},
        }
        mock_model.create_completion.return_value = mock_output
        result = generator.run("Test prompt")
        assert result["replies"] == ["Generated text"]
        assert result["meta"] == [mock_output]

    def test_run_with_generation_kwargs(self, generator_mock):
        """
        Test that a valid prompt and generation kwargs returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "choices": [{"text": "Generated text"}],
            "metadata": {"other_info": "Some metadata"},
        }
        mock_model.create_completion.return_value = mock_output
        generation_kwargs = {"max_tokens": 128}
        result = generator.run("Test prompt", generation_kwargs)
        assert result["replies"] == ["Generated text"]
        assert result["meta"] == [mock_output]

    @pytest.mark.integration
    def test_run(self, generator):
        """
        Test that a valid prompt returns a list of replies.
        """
        questions_and_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        for question, answer in questions_and_answers:
            prompt = f"GPT4 Correct User: Answer in a single word. {question} <|end_of_turn|>\n GPT4 Correct Assistant:"
            result = generator.run(prompt)

            assert "replies" in result
            assert isinstance(result["replies"], list)
            assert len(result["replies"]) > 0
            assert answer.lower() in result["replies"][0].lower().strip()

    @pytest.mark.integration
    def test_run_rag_pipeline(self, generator):
        """
        Test that a valid prompt returns a list of replies.
        """
        prompt_template = """GPT4 Correct User: Answer the question in a single word. {{question}}
        Context:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        <|end_of_turn|>
        GPT4 Correct Assistant:
        """
        rag_pipeline = Pipeline()
        rag_pipeline.add_component(
            instance=InMemoryBM25Retriever(document_store=InMemoryDocumentStore(), top_k=1), name="retriever"
        )
        rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
        rag_pipeline.add_component(instance=generator, name="llm")
        rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("retriever", "answer_builder.documents")

        # Populate the document store
        documents = [
            Document(content="The capital of France is Paris."),
            Document(content="The capital of Canada is Ottawa."),
            Document(content="The capital of Ghana is Accra."),
        ]
        rag_pipeline.get_component("retriever").document_store.write_documents(documents)

        # Query and assert
        questions_and_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        for question, answer in questions_and_answers:
            result = rag_pipeline.run(
                {
                    "retriever": {"query": question},
                    "prompt_builder": {"question": question},
                    "llm": {"generation_kwargs": {"temperature": 0.1}},
                    "answer_builder": {"query": question},
                }
            )

            assert len(result["answer_builder"]["answers"]) == 1
            generated_answer = result["answer_builder"]["answers"][0]
            assert answer.lower() in generated_answer.data.lower()
            assert generated_answer.query == question
            assert hasattr(generated_answer, "documents")
            assert hasattr(generated_answer, "meta")
