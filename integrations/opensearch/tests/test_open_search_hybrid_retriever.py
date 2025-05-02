from unittest.mock import MagicMock, patch

from haystack import Pipeline
from haystack.components.joiners.document_joiner import JoinMode
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.open_search_hybrid_retriever import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_initialization_default_parameters():
    doc_store = OpenSearchDocumentStore()
    hybrid_retriever = OpenSearchHybridRetriever(doc_store)
    
    assert hybrid_retriever.document_store == doc_store
    assert hybrid_retriever.text_embedder_kwargs == {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "progress_bar": False
    }
    assert hybrid_retriever.bm25_retriever_kwargs == {"top_k": 10, "document_store": doc_store}
    assert hybrid_retriever.embedding_retriever_kwargs == {"top_k": 10, "document_store": doc_store}
    assert hybrid_retriever.document_joiner_kwargs == {"join_mode": "concatenate"}
    assert hybrid_retriever.chat_prompt_builder_kwargs != {}
    assert hybrid_retriever.generator_kwargs == {}
    assert hybrid_retriever.answer_builder_kwargs == {}


def test_initialization_with_custom_parameters():
    doc_store = OpenSearchDocumentStore()
    custom_kwargs = {
        "text_embedder_kwargs": {"model": "custom-model", "progress_bar": True},
        "bm25_retriever_kwargs": {"top_k": 5},
        "embedding_retriever_kwargs": {"top_k": 5},
        "document_joiner_kwargs": {"join_mode": JoinMode.MERGE},
        "chat_prompt_builder_kwargs": {},
        "generator_kwargs": {"model": "gpt-4"},
        "answer_builder_kwargs": {"pattern": "custom"}
    }
    hybrid_retriever = OpenSearchHybridRetriever(doc_store, **custom_kwargs)

    # Verify the chat prompt template content
    template = hybrid_retriever.chat_prompt_builder_kwargs["template"]
    assert len(template) == 2
    
    # Check system message content
    system_message = template[0]
    assert isinstance(system_message, ChatMessage)
    assert system_message.is_from("system")
    assert "helpful AI assistant" in system_message.text

    # Check user message content
    user_message = template[1]
    assert isinstance(user_message, ChatMessage)
    assert user_message.is_from("user")
    assert "Context:" in user_message.text

    # Verify other parameters
    assert hybrid_retriever.text_embedder_kwargs == custom_kwargs["text_embedder_kwargs"]
    assert hybrid_retriever.bm25_retriever_kwargs == {"top_k": 5, "document_store": doc_store}
    assert hybrid_retriever.embedding_retriever_kwargs == {"top_k": 5, "document_store": doc_store}
    assert hybrid_retriever.document_joiner_kwargs == {"join_mode": JoinMode.MERGE}
    assert hybrid_retriever.generator_kwargs == {"model": "gpt-4"}
    assert hybrid_retriever.answer_builder_kwargs == {"pattern": "custom"}


def test_serialization():
    doc_store = OpenSearchDocumentStore()
    hybrid_retriever = OpenSearchHybridRetriever(doc_store)

    # to_dict
    retriever_dict = hybrid_retriever.to_dict()
    assert "document_store" in retriever_dict
    assert "text_embedder_kwargs" in retriever_dict
    assert "bm25_retriever_kwargs" in retriever_dict
    assert "embedding_retriever_kwargs" in retriever_dict
    assert "document_joiner_kwargs" in retriever_dict
    assert "chat_prompt_builder_kwargs" in retriever_dict
    assert "generator_kwargs" in retriever_dict
    assert "answer_builder_kwargs" in retriever_dict
    
    # from_dict
    new_retriever = OpenSearchHybridRetriever.from_dict(retriever_dict)
    assert isinstance(new_retriever, OpenSearchHybridRetriever)
    assert new_retriever.text_embedder_kwargs == hybrid_retriever.text_embedder_kwargs
    assert new_retriever.bm25_retriever_kwargs == hybrid_retriever.bm25_retriever_kwargs
    assert new_retriever.embedding_retriever_kwargs == hybrid_retriever.embedding_retriever_kwargs
    assert new_retriever.document_joiner_kwargs == hybrid_retriever.document_joiner_kwargs
    assert new_retriever.generator_kwargs == hybrid_retriever.generator_kwargs
    assert new_retriever.answer_builder_kwargs == hybrid_retriever.answer_builder_kwargs
    assert new_retriever.pipeline == hybrid_retriever.pipeline


@patch("haystack.components.embedders.sentence_transformers.SentenceTransformersTextEmbedder")
@patch("haystack_integrations.components.retrievers.opensearch.OpenSearchEmbeddingRetriever")
@patch("haystack_integrations.components.retrievers.opensearch.OpenSearchBM25Retriever")
@patch("haystack.components.joiners.document_joiner.DocumentJoiner")
@patch("haystack.components.builders.chat_prompt_builder.ChatPromptBuilder")
@patch("haystack.components.generators.chat.openai.OpenAIChatGenerator")
@patch("haystack.components.builders.answer_builder.AnswerBuilder")
def test_pipeline_execution(
    mock_answer_builder,
    mock_generator,
    mock_prompt_builder,
    mock_document_joiner,
    mock_bm25_retriever,
    mock_embedding_retriever,
    mock_text_embedder
):
    # Setup mock components
    mock_text_embedder.return_value = MagicMock()
    mock_embedding_retriever.return_value = MagicMock()
    mock_bm25_retriever.return_value = MagicMock()
    mock_document_joiner.return_value = MagicMock()
    mock_prompt_builder.return_value = MagicMock()
    mock_generator.return_value = MagicMock()
    mock_answer_builder.return_value = MagicMock()
    
    # Setup mock document store
    doc_store = MagicMock(spec=OpenSearchDocumentStore)
    
    # Create hybrid retriever
    hybrid_retriever = OpenSearchHybridRetriever(doc_store)
    
    # Create pipeline
    pipeline = hybrid_retriever._create_pipeline()
    
    # Verify pipeline components were created with correct parameters
    mock_text_embedder.assert_called_once_with(**hybrid_retriever.text_embedder_kwargs)
    mock_embedding_retriever.assert_called_once_with(**hybrid_retriever.embedding_retriever_kwargs)
    mock_bm25_retriever.assert_called_once_with(**hybrid_retriever.bm25_retriever_kwargs)
    mock_document_joiner.assert_called_once_with(**hybrid_retriever.document_joiner_kwargs)
    mock_prompt_builder.assert_called_once_with(
        **hybrid_retriever.chat_prompt_builder_kwargs,
        required_variables=["question", "documents"]
    )
    mock_generator.assert_called_once_with(**hybrid_retriever.generator_kwargs)
    mock_answer_builder.assert_called_once_with(**hybrid_retriever.answer_builder_kwargs)