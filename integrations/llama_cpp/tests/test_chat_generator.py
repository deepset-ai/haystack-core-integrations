import json
import os
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline
from haystack.components.builders.dynamic_chat_prompt_builder import DynamicChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator


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


class TestLlamaCppChatGenerator:
    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210.Q3_K_S.gguf"
        )
        filename = "openchat-3.5-1210.Q3_K_S.gguf"

        # Download GGUF model from HuggingFace
        download_file(gguf_model_path, str(model_path / filename), capsys)

        model_path = str(model_path / filename)
        generator = LlamaCppChatGenerator(model=model_path, n_ctx=8192, n_batch=512)
        generator.warm_up()
        return generator

    @pytest.fixture
    def generator_mock(self):
        mock_model = MagicMock()
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=2048, n_batch=512)
        generator.model = mock_model
        return generator, mock_model

    def test_default_init(self):
        """
        Test default initialization parameters.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf")

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 0
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 0, "n_batch": 512}
        assert generator.generation_kwargs == {}

    def test_custom_init(self):
        """
        Test custom initialization parameters.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf",
            n_ctx=8192,
            n_batch=512,
        )

        assert generator.model_path == "test_model.gguf"
        assert generator.n_ctx == 8192
        assert generator.n_batch == 512
        assert generator.model_kwargs == {"model_path": "test_model.gguf", "n_ctx": 8192, "n_batch": 512}
        assert generator.generation_kwargs == {}

    def test_ignores_model_path_if_specified_in_model_kwargs(self):
        """
        Test that model_path is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf",
            n_ctx=8192,
            n_batch=512,
            model_kwargs={"model_path": "other_model.gguf"},
        )
        assert generator.model_kwargs["model_path"] == "other_model.gguf"

    def test_ignores_n_ctx_if_specified_in_model_kwargs(self):
        """
        Test that n_ctx is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=512, n_batch=512, model_kwargs={"n_ctx": 8192})
        assert generator.model_kwargs["n_ctx"] == 8192

    def test_ignores_n_batch_if_specified_in_model_kwargs(self):
        """
        Test that n_batch is ignored if already specified in model_kwargs.
        """
        generator = LlamaCppChatGenerator(
            model="test_model.gguf", n_ctx=8192, n_batch=512, model_kwargs={"n_batch": 1024}
        )
        assert generator.model_kwargs["n_batch"] == 1024

    def test_raises_error_without_warm_up(self):
        """
        Test that the generator raises an error if warm_up() is not called before running.
        """
        generator = LlamaCppChatGenerator(model="test_model.gguf", n_ctx=512, n_batch=512)
        with pytest.raises(RuntimeError):
            generator.run("What is the capital of China?")

    def test_run_with_empty_message(self, generator_mock):
        """
        Test that an empty message returns an empty list of replies.
        """
        generator, _ = generator_mock
        result = generator.run([])
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 0

    def test_run_with_valid_message(self, generator_mock):
        """
        Test that a valid message returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "id": "unique-id-123",
            "model": "Test Model Path",
            "created": 1715226164,
            "choices": [
                {"index": 0, "message": {"content": "Generated text", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 14, "completion_tokens": 57, "total_tokens": 71},
        }
        mock_model.create_chat_completion.return_value = mock_output
        result = generator.run(messages=[ChatMessage.from_system("Test")])
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)
        assert result["replies"][0].content == "Generated text"
        assert result["replies"][0].role == ChatRole.ASSISTANT

    def test_run_with_generation_kwargs(self, generator_mock):
        """
        Test that a valid message and generation kwargs returns a list of replies.
        """
        generator, mock_model = generator_mock
        mock_output = {
            "id": "unique-id-123",
            "model": "Test Model Path",
            "created": 1715226164,
            "choices": [
                {"index": 0, "message": {"content": "Generated text", "role": "assistant"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 14, "completion_tokens": 57, "total_tokens": 71},
        }
        mock_model.create_chat_completion.return_value = mock_output
        generation_kwargs = {"max_tokens": 128}
        result = generator.run([ChatMessage.from_system("Write a 200 word paragraph.")], generation_kwargs)
        assert result["replies"][0].content == "Generated text"
        assert result["replies"][0].meta["finish_reason"] == "length"

    @pytest.mark.integration
    def test_run(self, generator):
        """
        Test that a valid message returns a list of replies.
        """
        questions_and_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        for question, answer in questions_and_answers:
            chat_message = ChatMessage.from_system(
                f"GPT4 Correct User: Answer in a single word. {question} <|end_of_turn|>\n GPT4 Correct Assistant:"
            )
            result = generator.run([chat_message])

            assert "replies" in result
            assert isinstance(result["replies"], list)
            assert len(result["replies"]) > 0
            assert any(answer.lower() in reply.content.lower() for reply in result["replies"])

    @pytest.mark.integration
    def test_run_rag_pipeline(self, generator):
        """
        Test that a valid message returns a list of replies.
        """
        document_store = InMemoryDocumentStore()
        documents = [
            Document(content="There are over 7,000 languages spoken around the world today."),
            Document(
                content="Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors."
            ),
            Document(
                content="In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves."
            ),
        ]
        document_store.write_documents(documents=documents)

        pipeline = Pipeline()
        pipeline.add_component(
            instance=InMemoryBM25Retriever(document_store=document_store, top_k=1),
            name="retriever",
        )
        pipeline.add_component(
            instance=DynamicChatPromptBuilder(runtime_variables=["query", "documents"]), name="prompt_builder"
        )
        pipeline.add_component(instance=generator, name="llm")
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")

        question = "How many languages are there?"
        location = "Puerto Rico"
        system_message = ChatMessage.from_system(
            "You are a helpful assistant giving out valuable information to tourists."
        )
        messages = [
            system_message,
            ChatMessage.from_user(
                """
        Given these documents and given that I am currently in {{ location }}, answer the question.\nDocuments:
            {% for doc in documents %}
                {{ doc.content }}
            {% endfor %}

            \nQuestion: {{query}}
            \nAnswer:
        """
            ),
        ]
        question = "Can I see bioluminescent waves at my current location?"
        result = pipeline.run(
            data={
                "retriever": {"query": question},
                "prompt_builder": {
                    "template_variables": {"location": location},
                    "prompt_source": messages,
                    "query": question,
                },
            }
        )

        replies = result['llm']['replies']
        assert len(replies) > 0
        assert any("bioluminescent waves" in reply.content for reply in replies)
        assert all(reply.role == ChatRole.ASSISTANT for reply in replies)


class TestLlamaCppChatGeneratorFunctionCalls:
    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/meetkai/functionary-small-v2.4-GGUF/resolve/main/functionary-small-v2.4.Q4_0.gguf"
        )
        filename = "functionary-small-v2.4.Q4_0.gguf"
        download_file(gguf_model_path, str(model_path / filename), capsys)
        model_path = str(model_path / filename)
        hf_tokenizer_path = "meetkai/functionary-small-v2.4-GGUF"
        generator = LlamaCppChatGenerator(
            model=model_path,
            n_ctx=8192,
            n_batch=512,
            model_kwargs={
                "chat_format": "functionary-v2",
                "hf_tokenizer_path": hf_tokenizer_path,
            },
        )
        generator.warm_up()
        return generator

    @pytest.mark.integration
    def test_function_call_scenario(self, generator):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "username": {"type": "string", "description": "The username to retrieve information for."}
                        },
                        "required": ["username"],
                    },
                    "description": "Retrieves detailed information about a user.",
                },
            }
        ]
        tool_choice = {"type": "function", "function": {"name": "get_user_info"}}

        messages = [
            ChatMessage.from_user("Get information for user john_doe"),
        ]
        generation_kwargs = {"tools": tools, "tool_choice": tool_choice}

        response = generator.run(messages=messages, generation_kwargs=generation_kwargs)

        assert "tool_calls" in response["replies"][0].meta
        tool_calls = response["replies"][0].meta["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "get_user_info"
        assert "username" in json.loads(tool_calls[0]["function"]["arguments"])
        assert response["replies"][0].role == ChatRole.ASSISTANT
