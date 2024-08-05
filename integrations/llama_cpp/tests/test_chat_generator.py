import json
import os
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.llama_cpp.chat.chat_generator import (
    LlamaCppChatGenerator,
    _convert_message_to_llamacpp_format,
)


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


def test_convert_message_to_llamacpp_format():
    message = ChatMessage.from_system("You are good assistant")
    assert _convert_message_to_llamacpp_format(message) == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert _convert_message_to_llamacpp_format(message) == {"role": "user", "content": "I have a question"}

    message = ChatMessage.from_function("Function call", "function_name")
    assert _convert_message_to_llamacpp_format(message) == {
        "role": "function",
        "content": "Function call",
        "name": "function_name",
    }


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
                content="""Elephants have been observed to behave in a way that indicates a high
                level of self-awareness, such as recognizing themselves in mirrors."""
            ),
            Document(
                content="""In certain parts of the world, like the Maldives, Puerto Rico,
                and San Diego, you can witness the phenomenon of bioluminescent waves."""
            ),
        ]
        document_store.write_documents(documents=documents)

        pipeline = Pipeline()
        pipeline.add_component(
            instance=InMemoryBM25Retriever(document_store=document_store, top_k=1),
            name="retriever",
        )
        pipeline.add_component(instance=ChatPromptBuilder(variables=["query", "documents"]), name="prompt_builder")
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
                    "template": messages,
                    "query": question,
                },
            }
        )

        replies = result["llm"]["replies"]
        assert len(replies) > 0
        assert any("bioluminescent waves" in reply.content for reply in replies)
        assert all(reply.role == ChatRole.ASSISTANT for reply in replies)

    @pytest.mark.integration
    def test_json_constraining(self, generator):
        """
        Test that the generator can output valid JSON.
        """
        messages = [ChatMessage.from_system("Output valid json only. List 2 people with their name and age.")]
        json_schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["people"],
        }

        result = generator.run(
            messages=messages,
            generation_kwargs={
                "response_format": {"type": "json_object", "schema": json_schema},
            },
        )

        assert "replies" in result
        assert isinstance(result["replies"], list)
        assert len(result["replies"]) > 0
        assert all(reply.role == ChatRole.ASSISTANT for reply in result["replies"])
        for reply in result["replies"]:
            assert json.loads(reply.content)
            assert isinstance(json.loads(reply.content), dict)
            assert "people" in json.loads(reply.content)
            assert isinstance(json.loads(reply.content)["people"], list)
            assert all(isinstance(person, dict) for person in json.loads(reply.content)["people"])
            assert all("name" in person for person in json.loads(reply.content)["people"])
            assert all("age" in person for person in json.loads(reply.content)["people"])
            assert all(isinstance(person["name"], str) for person in json.loads(reply.content)["people"])
            assert all(isinstance(person["age"], int) for person in json.loads(reply.content)["people"])


class TestLlamaCppChatGeneratorFunctionary:
    def get_current_temperature(self, location):
        """Get the current temperature in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
        elif "san francisco" in location.lower():
            return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

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
    def test_function_call(self, generator):
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
        response = generator.run(messages=messages, generation_kwargs={"tools": tools, "tool_choice": tool_choice})

        assert "tool_calls" in response["replies"][0].meta
        tool_calls = response["replies"][0].meta["tool_calls"]
        assert len(tool_calls) > 0
        assert tool_calls[0]["function"]["name"] == "get_user_info"
        assert "username" in json.loads(tool_calls[0]["function"]["arguments"])
        assert response["replies"][0].role == ChatRole.ASSISTANT

    def test_function_call_and_execute(self, generator):
        messages = [ChatMessage.from_user("What's the weather like in San Francisco?")]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Get the current temperature in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = generator.run(messages=messages, generation_kwargs={"tools": tools})

        available_functions = {
            "get_current_temperature": self.get_current_temperature,
        }

        assert "replies" in response
        assert len(response["replies"]) > 0

        first_reply = response["replies"][0]
        assert "tool_calls" in first_reply.meta
        tool_calls = first_reply.meta["tool_calls"]

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            assert function_name in available_functions
            function_response = available_functions[function_name](**function_args)
            function_message = ChatMessage.from_function(function_response, function_name)
            messages.append(function_message)

        second_response = generator.run(messages=messages)
        assert "replies" in second_response
        assert len(second_response["replies"]) > 0
        assert any("San Francisco" in reply.content for reply in second_response["replies"])
        assert any("72" in reply.content for reply in second_response["replies"])


class TestLlamaCppChatGeneratorChatML:

    @pytest.fixture
    def generator(self, model_path, capsys):
        gguf_model_path = (
            "https://huggingface.co/TheBloke/openchat-3.5-1210-GGUF/resolve/main/openchat-3.5-1210.Q3_K_S.gguf"
        )
        filename = "openchat-3.5-1210.Q3_K_S.gguf"
        download_file(gguf_model_path, str(model_path / filename), capsys)
        model_path = str(model_path / filename)
        generator = LlamaCppChatGenerator(
            model=model_path,
            n_ctx=8192,
            n_batch=512,
            model_kwargs={
                "chat_format": "chatml-function-calling",
            },
        )
        generator.warm_up()
        return generator

    @pytest.mark.integration
    def test_function_call_chatml(self, generator):
        messages = [
            ChatMessage.from_system(
                """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful,
                detailed, and polite answers to the user's questions. The assistant calls functions with appropriate
                input when necessary"""
            ),
            ChatMessage.from_user("Extract Jason is 25 years old"),
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "UserDetail",
                    "parameters": {
                        "type": "object",
                        "title": "UserDetail",
                        "properties": {
                            "name": {"title": "Name", "type": "string"},
                            "age": {"title": "Age", "type": "integer"},
                        },
                        "required": ["name", "age"],
                    },
                },
            }
        ]

        tool_choice = {"type": "function", "function": {"name": "UserDetail"}}

        response = generator.run(messages=messages, generation_kwargs={"tools": tools, "tool_choice": tool_choice})
        for reply in response["replies"]:
            assert "tool_calls" in reply.meta
            tool_calls = reply.meta["tool_calls"]
            assert len(tool_calls) > 0
            assert tool_calls[0]["function"]["name"] == "UserDetail"
            assert "name" in json.loads(tool_calls[0]["function"]["arguments"])
            assert "age" in json.loads(tool_calls[0]["function"]["arguments"])
            assert "Jason" in json.loads(tool_calls[0]["function"]["arguments"])["name"]
            assert 25 == json.loads(tool_calls[0]["function"]["arguments"])["age"]
