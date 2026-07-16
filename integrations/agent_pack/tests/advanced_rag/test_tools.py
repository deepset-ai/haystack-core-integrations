from typing import ClassVar

import pytest
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool, PipelineTool

from haystack_integrations.agent_pack.advanced_rag.tools import (
    DocumentStoreToolset,
    FetchDocumentsByFilterTool,
    GetMetadataFieldRangeTool,
    GetMetadataFieldValuesTool,
    ListMetadataFieldsTool,
    _accumulate_documents,
    _accumulate_pipeline_documents,
    _format_fetch_result,
    _format_pipeline_result,
    _format_retrieved_documents,
    _order_documents_for_reading,
    make_retrieval_pipeline_tool,
    make_retriever_tool,
)

DOCS = [
    Document(content="CRISPR gene editing breakthrough", meta={"category": "science", "year": 2021, "rating": 4.5}),
    Document(content="The fall of the Berlin Wall", meta={"category": "history", "year": 1989, "rating": 4.9}),
    Document(content="Quantum computing milestones", meta={"category": "science", "year": 2019}),
]

SPLIT_DOCS = [
    Document(content="b2", meta={"file_name": "b.pdf", "split_id": 2, "category": "report"}),
    Document(content="a1", meta={"file_name": "a.pdf", "split_id": 1, "category": "report"}),
    Document(content="b1", meta={"file_name": "b.pdf", "split_id": 1, "category": "report"}),
    Document(content="a0", meta={"file_name": "a.pdf", "split_id": 0, "category": "report"}),
]

# Reading order groups by file; group order follows first appearance, so both are acceptable.
READING_ORDERS = (["a0", "a1", "b1", "b2"], ["b1", "b2", "a0", "a1"])


@pytest.fixture
def store():
    document_store = InMemoryDocumentStore()
    document_store.write_documents(DOCS)
    return document_store


class _StoreWithoutMetadataMethods:
    pass


class TestListMetadataFieldsTool:
    def test_lists_fields_and_types(self, store):
        out = ListMetadataFieldsTool(store).invoke()
        assert "- category (keyword)" in out
        assert "- rating (float)" in out
        assert "- year (int)" in out
        assert "meta." in out  # instructs the prefix

    def test_empty_store(self):
        out = ListMetadataFieldsTool(InMemoryDocumentStore()).invoke()
        assert out == "No metadata fields found (the document store may be empty)."


class TestGetMetadataFieldValuesTool:
    def test_lists_unique_values(self, store):
        out = GetMetadataFieldValuesTool(store).invoke(field="category")
        assert out == "Field 'category' has 2 unique values: history, science"

    def test_strips_meta_prefix(self, store):
        tool = GetMetadataFieldValuesTool(store)
        assert tool.invoke(field="meta.category") == tool.invoke(field="category")

    def test_reports_missing_field(self, store):
        out = GetMetadataFieldValuesTool(store).invoke(field="nope")
        assert out == "Field 'nope' has no values in the document store."

    def test_caps_the_listing(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents([Document(content=f"d{i}", meta={"tag": f"tag-{i:03d}"}) for i in range(105)])
        out = GetMetadataFieldValuesTool(document_store).invoke(field="tag")
        assert "105 unique values" in out
        assert "… and 5 more" in out

    @pytest.mark.parametrize(
        "case",
        [
            ([], 0, False, "Field 'tag' has no values in the document store."),
            (["a", "b"], 2, False, "Field 'tag' has 2 unique values: a, b"),
            (["a", "b"], 5, False, "Field 'tag' has 5 unique values: a, b … and 3 more"),
            (["a"], 1, True, "Field 'tag' has 1 unique values: a … more values exist"),
        ],
    )
    def test_format_field_values(self, store, case):
        values, count, has_more_values, expected = case
        out = GetMetadataFieldValuesTool(store)._format_field_values(
            "tag", values, count, has_more_values=has_more_values
        )
        assert out == expected

    def test_forwards_size_and_reports_cursor_pagination(self):
        captured = {}

        class _CursorStore(InMemoryDocumentStore):
            def get_metadata_field_unique_values(self, metadata_field, search_term=None, size=10000, after=None):  # noqa: ARG002
                captured["size"] = size
                return (["a", "b"], {"after_key": "b"})  # a cursor dict: more pages exist

        out = GetMetadataFieldValuesTool(_CursorStore()).invoke(field="tag")
        assert captured["size"] == GetMetadataFieldValuesTool._MAX_LISTED_VALUES
        assert "more values exist" in out


class TestGetMetadataFieldRangeTool:
    def test_returns_min_and_max(self, store):
        out = GetMetadataFieldRangeTool(store).invoke(field="meta.year")
        assert out == "Field 'year': min=1989, max=2021"

    def test_reports_missing_field(self, store):
        out = GetMetadataFieldRangeTool(store).invoke(field="nope")
        assert out == "Field 'nope' has no numeric or orderable values in the document store."


class TestMetadataToolContract:
    """Behavior shared by all three metadata inspection tools."""

    TOOL_CLASSES: ClassVar[list[type]] = [ListMetadataFieldsTool, GetMetadataFieldValuesTool, GetMetadataFieldRangeTool]

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_rejects_store_without_the_required_method(self, tool_cls):
        with pytest.raises(ValueError, match="does not implement"):
            tool_cls(_StoreWithoutMetadataMethods())

    @pytest.mark.parametrize("tool_cls", TOOL_CLASSES)
    def test_serialization_roundtrip(self, store, tool_cls):
        tool = tool_cls(store)
        data = tool.to_dict()
        assert data["type"].endswith(tool_cls.__name__)

        restored = tool_cls.from_dict(data)
        assert isinstance(restored, tool_cls)
        assert restored.name == tool.name
        assert isinstance(restored.document_store, InMemoryDocumentStore)


class TestDocumentFormatting:
    def test_without_documents(self):
        assert "No documents matched" in _format_retrieved_documents([])

    def test_uses_short_refs_and_meta(self):
        doc = DOCS[0]
        out = _format_retrieved_documents([doc])
        assert f"[doc {doc.id[:8]}]" in out
        assert doc.id not in out  # the full 64-char id is not shown
        assert '"category": "science"' in out
        assert "CRISPR gene editing breakthrough" in out

    def test_pipeline_result_finds_the_document_list(self):
        result = {"other": 1, "documents": [DOCS[0]]}
        assert f"[doc {DOCS[0].id[:8]}]" in _format_pipeline_result(result)

    def test_pipeline_result_handles_empty_list_and_no_documents(self):
        assert "No documents matched" in _format_pipeline_result({"documents": []})
        assert _format_pipeline_result({"other": "value"}) == "{'other': 'value'}"

    def test_fetch_result_notes_extra_matches(self):
        out = _format_fetch_result({"documents": [DOCS[0]], "total_matched": 5})
        assert "… and 4 more documents matched; narrow the filter." in out


class TestReadingOrder:
    def test_groups_by_file_and_sorts_by_split(self):
        ordered = [d.content for d in _order_documents_for_reading(SPLIT_DOCS)]
        assert ordered in READING_ORDERS

    def test_sorts_without_group_field(self):
        docs = [Document(content="p3", meta={"page_number": 3}), Document(content="p1", meta={"page_number": 1})]
        assert [d.content for d in _order_documents_for_reading(docs)] == ["p1", "p3"]

    def test_keeps_order_without_known_fields(self):
        docs = [Document(content="one"), Document(content="two")]
        assert _order_documents_for_reading(docs) == docs


class TestDocumentAccumulation:
    """The `outputs_to_state` handlers that collect retrieved documents into the agent's State."""

    def test_deduplicates_by_id(self):
        acc = _accumulate_documents(None, DOCS[:2])
        acc = _accumulate_documents(acc, DOCS[1:])
        assert [d.id for d in acc] == [d.id for d in DOCS]

    def test_pipeline_handler_extracts_the_document_list(self):
        assert _accumulate_pipeline_documents(None, {"documents": DOCS[:1]}) == DOCS[:1]
        assert _accumulate_pipeline_documents(DOCS[:1], {"other": "value"}) == DOCS[:1]


class TestFetchDocumentsByFilterTool:
    def test_returns_documents_in_reading_order(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents(SPLIT_DOCS)
        result = FetchDocumentsByFilterTool(document_store).invoke(
            filters={"field": "meta.category", "operator": "==", "value": "report"}
        )
        assert [d.content for d in result["documents"]] in READING_ORDERS
        assert result["total_matched"] == 4

    def test_zero_matches(self, store):
        result = FetchDocumentsByFilterTool(store).invoke(
            filters={"field": "meta.category", "operator": "==", "value": "nonexistent"}
        )
        assert result == {"documents": [], "total_matched": 0}
        assert "No documents matched" in _format_fetch_result(result)

    def test_respects_and_clamps_max_docs(self, store):
        tool = FetchDocumentsByFilterTool(store, max_docs=2)
        science = {"field": "meta.category", "operator": "==", "value": "science"}
        assert len(tool.invoke(filters=science, max_docs=1)["documents"]) == 1
        assert len(tool.invoke(filters=science, max_docs=99)["documents"]) == 2  # clamped to the ceiling
        assert tool.parameters["properties"]["max_docs"]["maximum"] == 2

    def test_refuses_over_broad_filters(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents([Document(content=f"d{i}", meta={"category": "bulk"}) for i in range(25)])
        bulk = {"field": "meta.category", "operator": "==", "value": "bulk"}
        with pytest.raises(ValueError, match="matches 25 documents"):
            FetchDocumentsByFilterTool(document_store, max_docs=2)._fetch_documents(filters=bulk)
        # A higher fetch factor raises the refusal threshold (2 * 20 >= 25).
        result = FetchDocumentsByFilterTool(document_store, max_docs=2, max_fetch_factor=20)._fetch_documents(
            filters=bulk
        )
        assert result["total_matched"] == 25

    def test_serialization_roundtrip(self, store):
        tool = FetchDocumentsByFilterTool(store, max_docs=7, max_fetch_factor=3)
        restored = FetchDocumentsByFilterTool.from_dict(tool.to_dict())
        assert restored.max_docs == 7
        assert restored.max_fetch_factor == 3
        assert restored.name == "fetch_documents_by_filter"
        assert isinstance(restored.document_store, InMemoryDocumentStore)


class TestMakeRetrieverTool:
    def test_configures_component_tool(self, store):
        tool = make_retriever_tool(retriever=InMemoryBM25Retriever(document_store=store, top_k=3))
        assert isinstance(tool, ComponentTool)
        assert tool.name == "search_documents"
        assert set(tool.parameters["properties"]) == {"query", "filters"}
        assert tool.parameters["required"] == ["query"]
        assert "meta." in tool.parameters["properties"]["filters"]["description"]  # the filter grammar
        assert tool.outputs_to_state["documents"]["handler"] is _accumulate_documents

    def test_rejects_embedding_retrievers(self, store):
        with pytest.raises(ValueError, match="TextEmbeddingRetriever"):
            make_retriever_tool(retriever=InMemoryEmbeddingRetriever(document_store=store))


class TestMakeRetrievalPipelineTool:
    def test_builds_pipeline_tool(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        tool = make_retrieval_pipeline_tool(
            pipeline=pipeline,
            input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]},
            output_mapping={"retriever.documents": "documents"},
        )
        assert isinstance(tool, PipelineTool)
        assert tool.name == "search_documents"
        assert set(tool.parameters["properties"]) == {"query", "filters"}

    def test_requires_query_and_filters_mapping(self):
        with pytest.raises(ValueError, match='exactly the keys "query" and "filters"'):
            make_retrieval_pipeline_tool(pipeline=Pipeline(), input_mapping={"query": ["retriever.query"]})


class TestDocumentStoreToolset:
    def test_bundles_the_store_tools(self, store):
        toolset = DocumentStoreToolset(store, max_fetched_docs=3)
        assert [t.name for t in toolset] == [
            "list_metadata_fields",
            "get_metadata_field_values",
            "get_metadata_field_range",
            "fetch_documents_by_filter",
        ]
        assert toolset[3].max_docs == 3

    def test_serialization_roundtrip(self, store):
        toolset = DocumentStoreToolset(store, max_fetched_docs=3)
        data = toolset.to_dict()
        assert data["type"].endswith("DocumentStoreToolset")

        restored = DocumentStoreToolset.from_dict(data)
        assert isinstance(restored, DocumentStoreToolset)
        assert restored.max_fetched_docs == 3
        assert len(restored) == 4
