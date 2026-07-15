# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# The tools the advanced RAG agent uses.

import contextlib
import inspect
import json
from typing import Any

from haystack import Document, Pipeline, logging
from haystack.components.rankers import MetaFieldGroupingRanker
from haystack.components.retrievers.types import TextRetriever
from haystack.core.serialization import generate_qualified_class_name
from haystack.document_stores.types import DocumentStore
from haystack.tools import ComponentTool, PipelineTool, Tool, Toolset
from haystack.utils.deserialization import deserialize_component_inplace

from haystack_integrations.agent_pack.advanced_rag import prompts

logger = logging.getLogger(__name__)


# Shared input schema for the retrieval tool. The property names MUST match the retriever's input sockets (component
# path) / the input_mapping keys (pipeline path). The filter grammar is placed in the `filters` description so the LLM
# can learn it from the tool spec.
_RETRIEVAL_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": ("The search query: a short phrase or a few keywords describing the documents to retrieve."),
        },
        "filters": {"type": "object", "description": prompts.FILTER_GRAMMAR, "additionalProperties": True},
    },
    "required": ["query"],
}


def _accumulate_documents(current: list[Document] | None, new: list[Document]) -> list[Document]:
    """
    State handler: merge newly retrieved documents into the accumulated list, deduplicated by id.

    :param current: The documents accumulated so far (insertion order preserved).
    :param new: The documents from the latest retrieval call.
    :returns: The merged, deduplicated list.
    """
    current = current or []
    seen = {d.id for d in current}
    merged = list(current)
    for doc in new:
        if doc.id not in seen:
            seen.add(doc.id)
            merged.append(doc)
    return merged


def _accumulate_pipeline_documents(current: list[Document] | None, result: dict) -> list[Document]:
    """
    State handler for the pipeline tool: find the documents in the pipeline result and accumulate them.

    :param current: The documents accumulated so far.
    :param result: The pipeline result, as mapped by `output_mapping`.
    :returns: The merged, deduplicated list.
    """
    for value in result.values():
        if isinstance(value, list) and value and isinstance(value[0], Document):
            return _accumulate_documents(current, value)
    return current or []


def _format_retrieved_documents(documents: list[Document]) -> str:
    """
    Format retrieved documents as the tool-result string: short doc reference + metadata + content.

    :param documents: The retrieved documents.
    :returns: One block per document, or a message nudging the agent to relax the filters when empty.
    """
    # Length of the short document reference the LLM sees and cites (a prefix of `Document.id`).
    doc_ref_len = 8

    if not documents:
        return (
            "No documents matched. Try relaxing or removing the filters, or verify the filter "
            "values with the metadata tools."
        )
    blocks = []
    for doc in documents:
        content = (doc.content or "").strip()
        blocks.append(f"[doc {doc.id[:doc_ref_len]}]\n  meta: {json.dumps(doc.meta, default=str)}\n  {content}")
    logger.info("retrieval -> {count} documents", count=len(documents))
    return "\n".join(blocks)


def _format_pipeline_result(result: dict) -> str:
    """
    Format a retrieval pipeline result: find the documents in the output and format them.

    Tolerant of whatever the user's `output_mapping` produces. It uses the first output value that is a list of
    `Document` objects.

    :param result: The pipeline result, as mapped by `output_mapping`.
    :returns: The formatted documents, or `str(result)` if no document list is found in the outputs.
    """
    saw_empty_list = False
    for value in result.values():
        if isinstance(value, list) and value and isinstance(value[0], Document):
            return _format_retrieved_documents(value)
        saw_empty_list = saw_empty_list or (isinstance(value, list) and not value)
    if saw_empty_list:
        return _format_retrieved_documents([])
    return str(result)


def make_retriever_tool(
    *, retriever: TextRetriever, name: str = "search_documents", description: str | None = None
) -> ComponentTool:
    """
    `search_documents` = ComponentTool over a retriever component that accepts `query` and `filters`.

    The tool exposes `query` plus an optional `filters` parameter whose description teaches the LLM the Haystack
    filter grammar, and accumulates every retrieved document into the agent's `documents` state key.

    :param retriever: A standalone retriever component (not added to a `Pipeline`) following the `TextRetriever`
        protocol, i.e. its `run` method accepts `query` and `filters`.
    :param name: The tool name the LLM sees.
    :param description: The tool description the LLM sees.
        Defaults to `haystack_integrations.agent_pack.advanced_rag.prompts.RETRIEVAL_TOOL_DESCRIPTION`.
    :returns: The retrieval tool.
    """
    sockets = getattr(retriever, "__haystack_input__", None)
    socket_names = set(sockets._sockets_dict.keys()) if sockets is not None else set()
    missing = {"query", "filters"} - socket_names
    if missing:
        msg = (
            f"{type(retriever).__name__} does not accept the required inputs {sorted(missing)}. "
            f"The retriever must take `query` and `filters`. Wrap embedding retrievers that take a "
            f"`query_embedding` in haystack's `TextEmbeddingRetriever(retriever=..., text_embedder=...)`, "
            f"or use `make_retrieval_pipeline_tool` for a custom retrieval pipeline."
        )
        raise ValueError(msg)
    return ComponentTool(
        component=retriever,
        name=name,
        description=description or prompts.RETRIEVAL_TOOL_DESCRIPTION,
        parameters=_RETRIEVAL_TOOL_PARAMS,
        outputs_to_string={"source": "documents", "handler": _format_retrieved_documents},
        outputs_to_state={"documents": {"source": "documents", "handler": _accumulate_documents}},
    )


def make_retrieval_pipeline_tool(
    *,
    pipeline: Pipeline,
    input_mapping: dict[str, list[str]],
    output_mapping: dict[str, str] | None = None,
    name: str = "search_documents",
    description: str | None = None,
) -> PipelineTool:
    """
    `search_documents` = PipelineTool over a custom retrieval pipeline.

    The tool exposes `query` plus an optional `filters` parameter whose description teaches the LLM the Haystack
    filter grammar, and accumulates every retrieved document into the agent's `documents` state key.

    :param pipeline: The retrieval pipeline to wrap (e.g. embedder -> retriever, or hybrid retrieval).
    :param input_mapping: Maps the tool inputs to pipeline input sockets. Must have exactly the keys "query" and
        "filters", e.g. `{"query": ["embedder.text"], "filters": ["retriever.filters"]}`.
    :param output_mapping: Optional map of pipeline output sockets to tool outputs,
        e.g. `{"retriever.documents": "documents"}`. Defaults to all pipeline outputs. It can stay unset because the
        tool-result formatter (`outputs_to_string`) scans the outputs for the first value that is a list of documents;
        set it when the pipeline exposes more than one document list, or to hide outputs the formatter shouldn't see.
    :param name: The tool name the LLM sees.
    :param description: The tool description the LLM sees.
        Defaults to `haystack_integrations.agent_pack.advanced_rag.prompts.RETRIEVAL_TOOL_DESCRIPTION`.
    :returns: The retrieval tool.
    """
    if set(input_mapping) != {"query", "filters"}:
        msg = (
            '`input_mapping` must have exactly the keys "query" and "filters", '
            'e.g. {"query": ["embedder.text"], "filters": ["retriever.filters"]}.'
        )
        raise ValueError(msg)
    return PipelineTool(
        pipeline=pipeline,
        name=name,
        description=description or prompts.RETRIEVAL_TOOL_DESCRIPTION,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        parameters=_RETRIEVAL_TOOL_PARAMS,
        outputs_to_string={"handler": _format_pipeline_result},
        outputs_to_state={"documents": {"handler": _accumulate_pipeline_documents}},
    )


def _require_store_method(document_store: DocumentStore, method: str) -> None:
    """
    Raise if `document_store` does not implement `method` (the metadata methods are opt-in per store).

    :param document_store: The document store to check.
    :param method: The name of the method the store must implement.
    :raises ValueError: If the store does not implement the method.
    """
    if not hasattr(document_store, method):
        msg = (
            f"{type(document_store).__name__} does not implement `{method}`, which this tool requires. "
            f"Use a document store that supports metadata introspection."
        )
        raise ValueError(msg)


def _strip_meta_prefix(field: str) -> str:
    """
    Normalize a field name: filters use 'meta.year' but the store methods expect 'year'.

    :param field: The field name, with or without the 'meta.' prefix.
    :returns: The field name without the 'meta.' prefix.
    """
    return field.removeprefix("meta.")


class ListMetadataFieldsTool(Tool):
    """Tool that lists all metadata fields and their types from a document store."""

    def __init__(self, document_store: DocumentStore) -> None:
        """
        Create the tool.

        :param document_store: The document store to inspect. Must implement `get_metadata_fields_info`.
        :raises ValueError: If the store does not implement `get_metadata_fields_info`.
        """
        _require_store_method(document_store, "get_metadata_fields_info")
        self.document_store = document_store
        super().__init__(
            name="list_metadata_fields",
            description=(
                "Returns all metadata fields available on the documents and their types "
                "(e.g. keyword, int, float). Call this FIRST to learn what fields you can filter on. "
                "Returned field names do NOT include the 'meta.' prefix — add it when building a "
                "filter (field 'year' becomes 'meta.year')."
            ),
            parameters={"type": "object", "properties": {}},
            function=self._list_metadata_fields,
        )

    def _list_metadata_fields(self) -> str:
        """
        List the store's metadata fields and their types.

        :returns: One line per field with its type, or a note that no fields were found.
        """
        fields_info = self.document_store.get_metadata_fields_info()  # type: ignore[attr-defined]
        if not fields_info:
            return "No metadata fields found (the document store may be empty)."
        lines = [f"- {field} ({info.get('type', 'unknown')})" for field, info in sorted(fields_info.items())]
        return "Metadata fields (add the 'meta.' prefix when filtering):\n" + "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ListMetadataFieldsTool":
        """
        Deserialize the tool from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized tool.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="document_store")
        return cls(**inner_data)


class GetMetadataFieldValuesTool(Tool):
    """Tool that returns the distinct values of a metadata field from a document store."""

    # Cap on the number of values listed per call, protecting the agent's context window from high-cardinality fields
    # (the total count is always reported).
    _MAX_LISTED_VALUES = 100
    # How many values are fetched from the store to search over when the LLM passes a `search_term`.
    _SEARCH_FETCH_SIZE = 1000

    def __init__(self, document_store: DocumentStore) -> None:
        """
        Create the tool.

        :param document_store: The document store to inspect. Must implement `get_metadata_field_unique_values`.
        :raises ValueError: If the store does not implement `get_metadata_field_unique_values`.
        """
        _require_store_method(document_store, "get_metadata_field_unique_values")
        self.document_store = document_store
        super().__init__(
            name="get_metadata_field_values",
            description=(
                "Returns the distinct values of a metadata field. Use it before filtering on a "
                "keyword or boolean field, so your filter uses values that actually exist "
                "(filter values are matched exactly)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "The metadata field name, as returned by list_metadata_fields.",
                    },
                    "search_term": {
                        "type": "string",
                        "description": (
                            "Optional case-insensitive substring to narrow the returned values (e.g. "
                            "'beaut' matches 'All_Beauty'). Use it when a field has more values than "
                            "fit in one response."
                        ),
                    },
                },
                "required": ["field"],
            },
            function=self._get_metadata_field_values,
        )

    def _get_metadata_field_values(self, field: str, search_term: str | None = None) -> str:
        """
        List the distinct values of a metadata field, optionally narrowed by a substring search.

        The search is applied CLIENT-SIDE, on purpose: the stores' own `search_term` parameter is split roughly
        half/half between two incompatible semantics. OpenSearch, Elasticsearch, Weaviate, Chroma, pgvector and
        InMemory match it against document CONTENT and aggregate the surviving documents' values;
        ArcadeDB, Astra, Azure AI Search, MongoDB Atlas, Valkey and FalkorDB match the VALUES themselves
        (FalkorDB case-sensitively); Oracle matches both. Filtering the fetched values ourselves gives every store
        the same, unsurprising "substring of the value" semantics.

        :param field: The metadata field name, with or without the 'meta.' prefix.
        :param search_term: Optional case-insensitive substring the returned values must contain.
        :returns: The unique values of the field (listing capped for high-cardinality fields).
        """
        field = _strip_meta_prefix(field)
        method = self.document_store.get_metadata_field_unique_values  # type: ignore[attr-defined]
        signature_params = inspect.signature(method).parameters

        # Fetch enough values to list (or, when searching, to search over): without an explicit size, several stores
        # return as few as 10 values by default, while others would fetch thousands only for us to drop them. Most
        # paginating stores call the parameter `size` (Chroma, Weaviate, OpenSearch, Elasticsearch, ...); Qdrant
        # uses `limit`; some take neither and return everything (FAISS, AlloyDB, IBM DB).
        fetch_size = self._SEARCH_FETCH_SIZE if search_term is not None else self._MAX_LISTED_VALUES
        kwargs: dict[str, Any] = {}
        if "size" in signature_params:
            kwargs["size"] = fetch_size
        elif "limit" in signature_params:
            kwargs["limit"] = fetch_size

        result = method(field, **kwargs)
        # Return shapes: a bare list of values (FAISS, AlloyDB, IBM DB), a (values, total_count) tuple
        # (InMemory, Chroma, Weaviate, Pinecone, Astra, ...), or a (values, cursor) tuple from
        # cursor-paginating stores (OpenSearch, Elasticsearch, FalkorDB) — a non-None cursor dict in
        # place of the count means more pages exist.
        values = list(result[0]) if isinstance(result, tuple) else list(result)
        count = result[1] if isinstance(result, tuple) and isinstance(result[1], int) else len(values)
        has_more_values = (isinstance(result, tuple) and isinstance(result[1], dict)) or count > len(values)

        if search_term is None:
            return self._format_field_values(field, values, count, has_more_values=has_more_values)

        matching = [v for v in values if search_term.lower() in str(v).lower()]
        return self._format_searched_field_values(
            field, matching, search_term, searched=len(values), search_incomplete=has_more_values
        )

    def _format_field_values(self, field: str, values: list[Any], count: int, *, has_more_values: bool) -> str:
        """
        Format the unique values of a field as the tool-result string.

        :param field: The metadata field name (without the 'meta.' prefix).
        :param values: The unique values returned by the store.
        :param count: The total number of unique values (may exceed `len(values)`).
        :param has_more_values: Whether the store holds values beyond those returned.
        :returns: The listing (capped at `_MAX_LISTED_VALUES`), with a hint to narrow via search_term
            when values were left out, or a note when the field has no values.
        """
        if not values:
            return f"Field '{field}' has no values in the document store."
        shown = values[: self._MAX_LISTED_VALUES]
        listing = ", ".join(str(v) for v in shown)
        suffix = f" … and {count - len(shown)} more" if count > len(shown) else ""
        if has_more_values and count <= len(shown):
            suffix += " … more values exist"
        if count > len(shown) or has_more_values:
            suffix += "; use search_term to narrow them"
        return f"Field '{field}' has {count} unique values: {listing}{suffix}"

    def _format_searched_field_values(
        self, field: str, matching: list[Any], search_term: str, *, searched: int, search_incomplete: bool
    ) -> str:
        """
        Format the substring-matched values of a field as the tool-result string.

        :param field: The metadata field name (without the 'meta.' prefix).
        :param matching: The values containing the search term.
        :param search_term: The substring that was searched for.
        :param searched: How many stored values were searched.
        :param search_incomplete: Whether the store holds values beyond the ones searched.
        :returns: The matching values (listing capped at `_MAX_LISTED_VALUES`), with a note when the
            search covered only part of the stored values.
        """
        incomplete_note = (
            f" (only the first {searched} stored values were searched; more exist)" if search_incomplete else ""
        )
        if not matching:
            return f"Field '{field}' has no values containing '{search_term}'{incomplete_note}."
        shown = matching[: self._MAX_LISTED_VALUES]
        listing = ", ".join(str(v) for v in shown)
        suffix = f" … and {len(matching) - len(shown)} more" if len(matching) > len(shown) else ""
        return (
            f"Field '{field}' has {len(matching)} values containing '{search_term}'{incomplete_note}: {listing}{suffix}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetMetadataFieldValuesTool":
        """
        Deserialize the tool from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized tool.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="document_store")
        return cls(**inner_data)


class GetMetadataFieldRangeTool(Tool):
    """Tool that returns the minimum and maximum values of a metadata field from a document store."""

    def __init__(self, document_store: DocumentStore) -> None:
        """
        Create the tool.

        :param document_store: The document store to inspect. Must implement `get_metadata_field_min_max`.
        :raises ValueError: If the store does not implement `get_metadata_field_min_max`.
        """
        _require_store_method(document_store, "get_metadata_field_min_max")
        self.document_store = document_store
        super().__init__(
            name="get_metadata_field_range",
            description=(
                "Returns the minimum and maximum values of a metadata field. Use it before "
                "filtering on numeric fields (int, float) or orderable ones such as ISO dates."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "The metadata field name, as returned by list_metadata_fields.",
                    }
                },
                "required": ["field"],
            },
            function=self._get_metadata_field_range,
        )

    def _get_metadata_field_range(self, field: str) -> str:
        """
        Get the minimum and maximum values of a metadata field.

        :param field: The metadata field name, with or without the 'meta.' prefix.
        :returns: The min and max of the field, or a note that the field has no orderable values.
        """
        field = _strip_meta_prefix(field)
        result = self.document_store.get_metadata_field_min_max(field)  # type: ignore[attr-defined]
        if result.get("min") is None and result.get("max") is None:
            return f"Field '{field}' has no numeric or orderable values in the document store."
        return f"Field '{field}': min={result.get('min')}, max={result.get('max')}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GetMetadataFieldRangeTool":
        """
        Deserialize the tool from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized tool.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="document_store")
        return cls(**inner_data)


# Fields used to put filter-fetched documents into a sensible reading order: group the splits of the same parent file
# together, then sort them by their position in the file. The first field of each tuple that is present in the fetched
# documents' meta wins. Haystack's document splitters auto-add `split_id` (ordinal) and `split_idx_start` (character
# offset), some add `page_number` and `source_id`; converters add the file name/path.
_GROUP_FIELDS = ("file_name", "file_path", "source_id")
_SORT_FIELDS = ("split_id", "split_idx_start", "page_number")


def _order_documents_for_reading(documents: list[Document]) -> list[Document]:
    """
    Put filter-fetched documents into reading order.

    Filter-fetched documents carry no relevance order, so they are grouped by their parent file (first present field
    of `_GROUP_FIELDS`, via `MetaFieldGroupingRanker`) and sorted by their position within it (first present field of
    `_SORT_FIELDS`).

    :param documents: The fetched documents.
    :returns: The documents, grouped and sorted when the meta fields allow it.
    """
    group_by = next((f for f in _GROUP_FIELDS if any(f in d.meta for d in documents)), None)
    sort_by = next((f for f in _SORT_FIELDS if any(f in d.meta for d in documents)), None)
    if group_by:
        ranker = MetaFieldGroupingRanker(group_by=group_by, sort_docs_by=sort_by)
        return ranker.run(documents=documents)["documents"]
    if sort_by:
        # Sort by the field value, placing documents with a missing value last. Suppress the
        # TypeError raised on mixed, non-comparable values — keep the original order then.
        with contextlib.suppress(TypeError):
            return sorted(documents, key=lambda d: (sort_by not in d.meta, d.meta.get(sort_by)))
    return documents


def _format_fetch_result(result: dict[str, Any]) -> str:
    """
    Format a fetch-by-filter result: the documents plus a note when more matched than were shown.

    :param result: The result from `FetchDocumentsByFilterTool._fetch_documents`, containing `documents` and
        `total_matched`.
    :returns: The formatted documents, or a message nudging the agent to adjust the filter.
    """
    documents: list[Document] = result["documents"]
    total_matched: int = result["total_matched"]
    formatted = _format_retrieved_documents(documents)
    extra = total_matched - len(documents)
    if extra > 0:
        formatted += f"\n… and {extra} more documents matched; narrow the filter."
    return formatted


def _fetch_documents_tool_params(max_docs: int) -> dict[str, Any]:
    """
    Build the input schema for the fetch-by-filter tool: a (required) filter plus an optional doc cap.

    :param max_docs: The configured ceiling for `max_docs`, embedded in the schema.
    :returns: The JSON schema.
    """
    return {
        "type": "object",
        "properties": {
            "filters": {"type": "object", "description": prompts.FILTER_GRAMMAR, "additionalProperties": True},
            "max_docs": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_docs,
                "description": f"How many documents to return at most. Defaults to the maximum, {max_docs}.",
            },
        },
        "required": ["filters"],
    }


class FetchDocumentsByFilterTool(Tool):
    """
    Tool that fetches documents directly from a document store by metadata filter.

    Unlike a scored retrieval tool, this fetches without any relevance ranking, so an agent can grab specific documents
    (e.g. a known title or source file) without going through a relevance search. The fetched documents are put into
    reading order first: grouped by their parent file (`file_name`/`file_path`/`source_id`) and sorted by their
    position within it (`split_id`/`split_idx_start`/`page_number`), using whichever of those metadata fields the
    documents carry.
    """

    def __init__(self, document_store: DocumentStore, max_docs: int = 10, max_fetch_factor: int = 10) -> None:
        """
        Create the tool.

        :param document_store: The document store to fetch documents from.
        :param max_docs: Ceiling on the number of documents shown to the agent per fetch. Unlike scored retrieval, a
            filter fetch is not bounded by a retriever's `top_k`, so this caps the tool result instead. The LLM can
            request fewer via the tool's optional `max_docs` input, but never more.
        :param max_fetch_factor: How many times the effective document cap a filter may match before the fetch is
            refused outright (when the store supports `count_documents_by_filter`) — the refusal is surfaced to the
            LLM as an error it can recover from by narrowing the filter.
        """
        self.document_store = document_store
        self.max_docs = max_docs
        self.max_fetch_factor = max_fetch_factor
        super().__init__(
            name="fetch_documents_by_filter",
            description=prompts.FILTER_RETRIEVER_TOOL_DESCRIPTION,
            parameters=_fetch_documents_tool_params(max_docs),
            # We purposefully return a dict with `documents` and `total_matched`
            function=self._fetch_documents,
            # This converts the dict into a single string and expects both keys to be present
            outputs_to_string={"handler": _format_fetch_result},
            # This accumulates the documents into the agent's state
            outputs_to_state={"documents": {"source": "documents", "handler": _accumulate_documents}},
        )

    def _fetch_documents(self, filters: dict[str, Any], max_docs: int | None = None) -> dict[str, Any]:
        """
        Fetch the documents matching a metadata filter, in reading order.

        :param filters: The metadata filter, in Haystack filter syntax.
        :param max_docs: How many documents to return at most; clamped to the configured ceiling (`self.max_docs`),
            which is also the default.
        :returns: The ordered documents (capped) plus how many documents matched the filter overall. Formatted for
            the LLM by `_format_fetch_result`.
        :raises ValueError: If the filter matches far more documents than can be shown (see below). The `Agent`
            surfaces this to the LLM as an error tool message, so it can recover by narrowing the filter.
        """
        effective_max = max(1, min(max_docs, self.max_docs)) if max_docs is not None else self.max_docs
        # Guard the expensive fetch: `filter_documents` has no limit parameter, so on a broad filter a real database
        # could ship thousands of documents only for us to show `max_docs` of them. When the store supports the cheap
        # `count_documents_by_filter` aggregation (opt-in, like the metadata methods), refuse over-broad filters
        # without fetching anything.
        fetch_limit = self.max_fetch_factor * effective_max
        counter = getattr(self.document_store, "count_documents_by_filter", None)
        if counter is not None:
            count = counter(filters=filters)
            if count == 0:
                return {"documents": [], "total_matched": 0}
            if count > fetch_limit:
                msg = (
                    f"Filter matches {count} documents, but at most {effective_max} can be shown per fetch — retrieving"
                    f" that many would be slow and the result would be mostly truncated anyway, so nothing was fetched."
                    f" This tool is meant for grabbing a small, specific set of documents. Add more conditions to "
                    f"narrow the filter, or use a relevance-based search to get only the most relevant documents."
                )
                raise ValueError(msg)
        documents = _order_documents_for_reading(self.document_store.filter_documents(filters=filters))
        return {"documents": documents[:effective_max], "total_matched": len(documents)}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tool to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "document_store": self.document_store.to_dict(),
                "max_docs": self.max_docs,
                "max_fetch_factor": self.max_fetch_factor,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FetchDocumentsByFilterTool":
        """
        Deserialize the tool from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized tool.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="document_store")
        return cls(**inner_data)


class DocumentStoreToolset(Toolset):
    """
    All document-store-backed tools as one unit.

    Bundles the three metadata inspection tools (`ListMetadataFieldsTool`, `GetMetadataFieldValuesTool`,
    `GetMetadataFieldRangeTool`) and the direct `FetchDocumentsByFilterTool`, so they can be handed to an `Agent`
    (or combined with a retrieval tool) as a single object.
    """

    def __init__(self, document_store: DocumentStore, max_fetched_docs: int = 10) -> None:
        """
        Create the toolset.

        :param document_store: The document store all tools run against. Must implement the metadata introspection
            methods (`get_metadata_fields_info`, `get_metadata_field_unique_values`, `get_metadata_field_min_max`).
        :param max_fetched_docs: Maximum number of documents `fetch_documents_by_filter` shows per fetch (see
            `FetchDocumentsByFilterTool.max_docs`).
        """
        self.document_store = document_store
        self.max_fetched_docs = max_fetched_docs
        super().__init__(
            tools=[
                ListMetadataFieldsTool(document_store),
                GetMetadataFieldValuesTool(document_store),
                GetMetadataFieldRangeTool(document_store),
                FetchDocumentsByFilterTool(document_store, max_docs=max_fetched_docs),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the toolset to a dictionary."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"document_store": self.document_store.to_dict(), "max_fetched_docs": self.max_fetched_docs},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentStoreToolset":
        """
        Deserialize the toolset from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized toolset.
        """
        inner_data = data["data"]
        deserialize_component_inplace(inner_data, key="document_store")
        return cls(**inner_data)
