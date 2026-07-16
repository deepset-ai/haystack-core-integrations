from typing import ClassVar

import pytest
from haystack import Document, Pipeline, component
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
    _format_fetch_result,
    _format_retrieved_documents,
    _make_retrieval_pipeline_tool,
    _make_retriever_tool,
    _order_documents_for_reading,
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

    def test_accepts_meta_prefixed_field_names(self, store):
        # Prefix normalization is the document store's job; InMemoryDocumentStore handles both forms.
        tool = GetMetadataFieldValuesTool(store)
        assert tool.invoke(field="meta.category").endswith(": history, science")
        assert tool.invoke(field="category").endswith(": history, science")

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
            ([], 0, "Field 'tag' has no values in the document store."),
            (["a", "b"], 2, "Field 'tag' has 2 unique values: a, b"),
            (["a", "b"], 5, "Field 'tag' has 5 unique values: a, b … and 3 more"),
            (["a"], None, "Field 'tag' has at least 1 unique values: a … more values may exist"),
        ],
    )
    def test_format_field_values(self, store, case):
        values, total, expected = case
        out = GetMetadataFieldValuesTool(store)._format_field_values("tag", values, total)
        assert out == expected

    @pytest.mark.parametrize(
        "case",
        [
            # (values, total) tuple: the int is the store-computed total, trusted even beyond the page
            # (InMemory, Chroma, Weaviate, pgvector, Pinecone, Astra, ...).
            ((["a", "b"], 40), 100, (["a", "b"], 40)),
            # (values, cursor) tuple mid-pagination: more pages exist, the total is unknown. OpenSearch and
            # Elasticsearch return the composite aggregation's `after_key`, a dict keyed by the field name;
            # FalkorDB returns an offset dict.
            ((["a", "b"], {"tag": "b"}), 100, (["a", "b"], None)),
            ((["a", "b"], {"offset": 2}), 100, (["a", "b"], None)),
            # (values, cursor) tuple on the last page: a None cursor means the page is the complete set.
            ((["a", "b"], None), 100, (["a", "b"], 2)),
            # Bare list shorter than the requested page (Qdrant with few values): complete.
            (["a", "b"], 100, (["a", "b"], 2)),
            # Bare full page (Qdrant at its `limit`): possibly truncated, the total is unknown.
            (["a", "b"], 2, (["a", "b"], None)),
            # Bare list from stores without a size/limit parameter (FAISS, AlloyDB, IBM DB): complete.
            (["a", "b"], None, (["a", "b"], 2)),
        ],
    )
    def test_normalize_unique_values_result(self, case):
        result, requested_size, expected = case
        normalized = GetMetadataFieldValuesTool._normalize_unique_values_result(result, requested_size=requested_size)
        assert normalized == expected

    def test_forwards_size_and_reports_cursor_pagination(self):
        captured = {}

        class _CursorStore(InMemoryDocumentStore):
            def get_metadata_field_unique_values(self, metadata_field, search_term=None, size=10000, after=None):  # noqa: ARG002
                captured["size"] = size
                return (["a", "b"], {metadata_field: "b"})  # a field-keyed `after_key` cursor: more pages exist

        out = GetMetadataFieldValuesTool(_CursorStore()).invoke(field="tag")
        assert captured["size"] == GetMetadataFieldValuesTool._MAX_LISTED_VALUES
        assert out == "Field 'tag' has at least 2 unique values: a, b … more values may exist"

    def test_forwards_limit_and_treats_a_full_page_as_possibly_truncated(self):
        captured = {}

        class _TruncatingLimitStore(InMemoryDocumentStore):
            def get_metadata_field_unique_values(self, metadata_field, filters=None, limit=100, offset=0):  # noqa: ARG002
                captured["limit"] = limit
                return [f"v{i}" for i in range(limit)]  # Qdrant-style: silently cut off at `limit`

        out = GetMetadataFieldValuesTool(_TruncatingLimitStore()).invoke(field="tag")
        assert captured["limit"] == GetMetadataFieldValuesTool._MAX_LISTED_VALUES
        assert "has at least 100 unique values" in out
        assert "more values may exist" in out

    def test_bare_list_without_pagination_params_is_reported_as_complete(self):
        class _BareListStore(InMemoryDocumentStore):
            def get_metadata_field_unique_values(self, field_name):  # noqa: ARG002
                return ["x", "y"]  # FAISS-style: no size/limit parameter, the complete set

        out = GetMetadataFieldValuesTool(_BareListStore()).invoke(field="tag")
        assert out == "Field 'tag' has 2 unique values: x, y"

    def test_supplies_required_parameters_without_defaults(self):
        captured = {}

        class _NoDefaultsStore(InMemoryDocumentStore):
            def get_metadata_field_unique_values(self, metadata_field, search_term, from_, size):  # noqa: ARG002
                captured.update(search_term=search_term, from_=from_, size=size)
                return (["a"], 1)  # pgvector-style: all parameters are required

        out = GetMetadataFieldValuesTool(_NoDefaultsStore()).invoke(field="tag")
        assert captured == {"search_term": None, "from_": 0, "size": GetMetadataFieldValuesTool._MAX_LISTED_VALUES}
        assert out == "Field 'tag' has 1 unique values: a"


class TestGetMetadataFieldRangeTool:
    def test_returns_min_and_max(self, store):
        out = GetMetadataFieldRangeTool(store).invoke(field="meta.year")
        assert out == "Field 'meta.year': min=1989, max=2021"

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

    def test_fetch_result_notes_extra_matches_with_paging_guidance(self):
        out = _format_fetch_result({"documents": [DOCS[0]], "total_matched": 5, "offset": 0})
        assert "… and 4 more documents matched; call again with offset=1 to continue, or narrow the filter." in out

    def test_fetch_result_reports_an_offset_beyond_the_match_set(self):
        out = _format_fetch_result({"documents": [], "total_matched": 5, "offset": 10})
        assert out == "No documents at offset 10: only 5 documents match the filter."


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

    def test_keeps_order_on_mixed_sort_value_types(self):
        # int and str split_ids in the same group are not comparable; the TypeError is suppressed.
        docs = [
            Document(content="s1", meta={"file_name": "a.pdf", "split_id": "one"}),
            Document(content="s0", meta={"file_name": "a.pdf", "split_id": 0}),
        ]
        assert _order_documents_for_reading(docs) == docs


class TestDocumentAccumulation:
    """The `outputs_to_state` handlers that collect retrieved documents into the agent's State."""

    def test_deduplicates_by_id(self):
        acc = _accumulate_documents(None, DOCS[:2])
        acc = _accumulate_documents(acc, DOCS[1:])
        assert [d.id for d in acc] == [d.id for d in DOCS]


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
        assert result == {"documents": [], "total_matched": 0, "offset": 0}
        # The fetch tool requires a filter, so its empty-result message must not suggest removing it.
        assert _format_fetch_result(result) == (
            "No documents matched the filter. Verify the field names and values with the metadata tools."
        )

    def test_pages_through_matches_with_offset(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents(SPLIT_DOCS)
        tool = FetchDocumentsByFilterTool(document_store, max_docs=3)
        report = {"field": "meta.category", "operator": "==", "value": "report"}

        first = tool.invoke(filters=report)
        second = tool.invoke(filters=report, offset=3)

        # The two pages line up into the full match set: reading order is stable across calls.
        assert [d.content for d in first["documents"] + second["documents"]] in READING_ORDERS
        assert first["total_matched"] == second["total_matched"] == 4
        assert "call again with offset=3 to continue" in _format_fetch_result(first)
        assert "more documents matched" not in _format_fetch_result(second)

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
        tool = _make_retriever_tool(retriever=InMemoryBM25Retriever(document_store=store, top_k=3))
        assert isinstance(tool, ComponentTool)
        assert tool.name == "search_documents"
        assert set(tool.parameters["properties"]) == {"query", "filters"}
        assert tool.parameters["required"] == ["query"]
        assert "meta." in tool.parameters["properties"]["filters"]["description"]  # the filter grammar
        assert tool.outputs_to_state["documents"]["handler"] is _accumulate_documents

    def test_rejects_embedding_retrievers(self, store):
        with pytest.raises(ValueError, match="TextEmbeddingRetriever"):
            _make_retriever_tool(retriever=InMemoryEmbeddingRetriever(document_store=store))

    def test_rejects_retrievers_without_a_documents_output(self):
        @component
        class _NoDocumentsRetriever:
            @component.output_types(results=list)
            def run(self, query: str, filters: dict | None = None) -> dict:  # noqa: ARG002
                return {"results": []}

        with pytest.raises(ValueError, match="does not return a `documents` output"):
            _make_retriever_tool(retriever=_NoDocumentsRetriever())


class TestMakeRetrievalPipelineTool:
    def test_builds_pipeline_tool(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        tool = _make_retrieval_pipeline_tool(
            pipeline=pipeline,
            input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]},
            output_mapping={"retriever.documents": "documents"},
        )
        assert isinstance(tool, PipelineTool)
        assert tool.name == "search_documents"
        assert set(tool.parameters["properties"]) == {"query", "filters"}

    def test_requires_query_and_filters_mapping(self):
        with pytest.raises(ValueError, match='exactly the keys "query" and "filters"'):
            _make_retrieval_pipeline_tool(pipeline=Pipeline(), input_mapping={"query": ["retriever.query"]})

    def test_works_without_output_mapping_when_pipeline_exposes_documents(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        tool = _make_retrieval_pipeline_tool(
            pipeline=pipeline, input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]}
        )
        assert isinstance(tool, PipelineTool)

    def test_requires_a_documents_output_in_the_mapping(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        with pytest.raises(ValueError, match='map one of the pipeline outputs to "documents"'):
            _make_retrieval_pipeline_tool(
                pipeline=pipeline,
                input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]},
                output_mapping={"retriever.documents": "results"},
            )

    def test_requires_a_documents_output_socket_without_mapping(self, store):
        @component
        class _Renamer:
            @component.output_types(results=list)
            def run(self, documents: list) -> dict:
                return {"results": documents}

        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        pipeline.add_component("renamer", _Renamer())
        pipeline.connect("retriever.documents", "renamer.documents")
        with pytest.raises(ValueError, match="does not expose a `documents` output socket"):
            _make_retrieval_pipeline_tool(
                pipeline=pipeline, input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]}
            )


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
