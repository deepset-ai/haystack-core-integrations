# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import json
from typing import Any

from haystack import Document, Pipeline
from haystack.components.rankers import MetaFieldGroupingRanker
from haystack.components.retrievers.types import TextRetriever
from haystack.core.serialization import generate_qualified_class_name
from haystack.document_stores.types import DocumentStore
from haystack.tools import ComponentTool, PipelineTool, Tool, Toolset
from haystack.utils.deserialization import deserialize_component_inplace

from haystack_integrations.agent_pack.advanced_rag import prompts

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
    return "\n".join(blocks)


def _make_retriever_tool(
    *, retriever: TextRetriever, name: str = "search_documents", description: str | None = None
) -> ComponentTool:
    """
    `search_documents` = ComponentTool over a retriever component that accepts `query` and `filters`.

    The tool exposes `query` plus an optional `filters` parameter whose description teaches the LLM the Haystack
    filter grammar, and accumulates every retrieved document into the agent's `documents` state key.

    :param retriever: A standalone retriever component (not added to a `Pipeline`) following the `TextRetriever`
        protocol, i.e. its `run` method accepts `query` and `filters` and it returns a `documents` output.
    :param name: The tool name the LLM sees.
    :param description: The tool description the LLM sees. Defaults to a pre-made description of relevance
        search with optional metadata filtering.
    :returns: The retrieval tool.
    """
    input_sockets = getattr(retriever, "__haystack_input__", None)
    input_names = set(input_sockets._sockets_dict.keys()) if input_sockets is not None else set()
    missing = {"query", "filters"} - input_names
    if missing:
        msg = (
            f"{type(retriever).__name__} does not accept the required inputs {sorted(missing)}. "
            f"The retriever must take `query` and `filters`. Wrap embedding retrievers that take a "
            f"`query_embedding` in haystack's `TextEmbeddingRetriever(retriever=..., text_embedder=...)`, "
            f"or pass a retrieval `Pipeline` as the `retriever` instead."
        )
        raise ValueError(msg)
    output_sockets = getattr(retriever, "__haystack_output__", None)
    output_names = set(output_sockets._sockets_dict.keys()) if output_sockets is not None else set()
    if "documents" not in output_names:
        msg = (
            f"{type(retriever).__name__} does not return a `documents` output, which the tool needs to "
            f"collect the retrieved documents."
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


def _make_retrieval_pipeline_tool(
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
    filter grammar, and accumulates every retrieved document into the agent's `documents` state key. The wrapped
    pipeline must therefore end up with a `documents` tool output — either through a pipeline output socket named
    `documents`, or through `output_mapping`.

    :param pipeline: The retrieval pipeline to wrap (e.g. embedder -> retriever, or hybrid retrieval).
    :param input_mapping: Maps the tool inputs to pipeline input sockets. Must have exactly the keys "query" and
        "filters", e.g. `{"query": ["embedder.text"], "filters": ["retriever.filters"]}`.
    :param output_mapping: Optional map of pipeline output sockets to tool outputs,
        e.g. `{"retriever.documents": "documents"}`. Must map one output to "documents" when provided. Can stay unset
        when the pipeline itself exposes a `documents` output socket (each unmapped pipeline output becomes a tool
        output named after its socket).
    :param name: The tool name the LLM sees.
    :param description: The tool description the LLM sees. Defaults to a pre-made description of relevance
        search with optional metadata filtering.
    :returns: The retrieval tool.
    """
    if set(input_mapping) != {"query", "filters"}:
        msg = (
            '`input_mapping` must have exactly the keys "query" and "filters", '
            'e.g. {"query": ["embedder.text"], "filters": ["retriever.filters"]}.'
        )
        raise ValueError(msg)
    if output_mapping is not None:
        if "documents" not in output_mapping.values():
            msg = (
                '`output_mapping` must map one of the pipeline outputs to "documents", '
                'e.g. {"retriever.documents": "documents"} — the tool needs it to collect the retrieved documents.'
            )
            raise ValueError(msg)
    else:
        leaf_socket_names = {socket for sockets in pipeline.outputs().values() for socket in sockets}
        if "documents" not in leaf_socket_names:
            msg = (
                "The pipeline does not expose a `documents` output socket. Provide an `output_mapping` that maps "
                'one of the pipeline outputs to "documents", e.g. {"retriever.documents": "documents"} — the tool '
                "needs it to collect the retrieved documents."
            )
            raise ValueError(msg)
    return PipelineTool(
        pipeline=pipeline,
        name=name,
        description=description or prompts.RETRIEVAL_TOOL_DESCRIPTION,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        parameters=_RETRIEVAL_TOOL_PARAMS,
        outputs_to_string={"source": "documents", "handler": _format_retrieved_documents},
        outputs_to_state={"documents": {"source": "documents", "handler": _accumulate_documents}},
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
    # (the total count is reported alongside when the store provides one).
    _MAX_LISTED_VALUES = 100

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
                },
                "required": ["field"],
            },
            function=self._get_metadata_field_values,
        )

    def _get_metadata_field_values(self, field: str) -> str:
        """
        List the distinct values of a metadata field.

        :param field: The metadata field name, with or without the 'meta.' prefix (normalized by the document store).
        :returns: The unique values of the field (listing capped for high-cardinality fields).
        """
        method = self.document_store.get_metadata_field_unique_values  # type: ignore[attr-defined]
        signature_params = inspect.signature(method).parameters

        kwargs: dict[str, Any] = {}
        # pgvector declares `search_term` and `from_` without defaults, so pass their neutral values when present.
        if "search_term" in signature_params:
            kwargs["search_term"] = None
        if "from_" in signature_params:
            kwargs["from_"] = 0
        # Fetch up to the listing cap: without an explicit size, several stores return as few as 10 values by
        # default, while others would fetch thousands only for us to drop them. Most paginating stores call the
        # parameter `size` (Chroma, Weaviate, OpenSearch, Elasticsearch, ...); Qdrant calls it `limit`; some take
        # neither and return everything (FAISS, AlloyDB, IBM DB).
        requested_size: int | None = None
        if "size" in signature_params:
            kwargs["size"] = requested_size = self._MAX_LISTED_VALUES
        elif "limit" in signature_params:
            kwargs["limit"] = requested_size = self._MAX_LISTED_VALUES

        result = method(field, **kwargs)
        values, total = self._normalize_unique_values_result(result, requested_size=requested_size)
        return self._format_field_values(field, values, total)

    @staticmethod
    def _normalize_unique_values_result(result: Any, *, requested_size: int | None) -> tuple[list[Any], int | None]:
        """
        Normalize the return shapes of `get_metadata_field_unique_values` across document stores.

        The method is not standardized; three dialects exist:

        - A `(values, total)` tuple (InMemory, Chroma, Weaviate, pgvector, Pinecone, Astra, ...): the int is the
          total number of unique values, computed by the store independently of the requested page size.
        - A `(values, cursor)` tuple from cursor-paginating stores (OpenSearch, Elasticsearch, FalkorDB): a dict
          pointing at the next page sits in place of the count. A non-None cursor means more values exist and the
          total is unknown; None means the returned values are the complete set.
        - A bare list of values without any count. FAISS, AlloyDB and IBM DB return the complete value set, but
          Qdrant silently truncates to its `limit` parameter — so a full page (`len(values) == requested_size`)
          may be cut off and the total counts as unknown then.

        :param result: The raw return value of the store method.
        :param requested_size: The page size passed to the store's `size`/`limit` parameter, or None when the
            store takes neither and returns everything.
        :returns: The values, plus the total number of unique values — None when the store did not report a
            total and more values may exist beyond the returned ones.
        """
        if isinstance(result, tuple):
            values, second = list(result[0]), result[1]
            if isinstance(second, int):
                return values, second
            return values, len(values) if second is None else None
        values = list(result)
        if requested_size is not None and len(values) >= requested_size:
            return values, None
        return values, len(values)

    def _format_field_values(self, field: str, values: list[Any], total: int | None) -> str:
        """
        Format the unique values of a field as the tool-result string.

        :param field: The metadata field name (without the 'meta.' prefix).
        :param values: The unique values returned by the store.
        :param total: The total number of unique values (may exceed `len(values)`), or None when the store did
            not report a total and more values may exist.
        :returns: The listing (capped at `_MAX_LISTED_VALUES`), or a note when the field has no values.
        """
        if not values:
            return f"Field '{field}' has no values in the document store."
        shown = values[: self._MAX_LISTED_VALUES]
        listing = ", ".join(str(v) for v in shown)
        if total is None:
            return f"Field '{field}' has at least {len(shown)} unique values: {listing} … more values may exist"
        suffix = f" … and {total - len(shown)} more" if total > len(shown) else ""
        return f"Field '{field}' has {total} unique values: {listing}{suffix}"

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

        :param field: The metadata field name, with or without the 'meta.' prefix (normalized by the document store).
        :returns: The min and max of the field, or a note that the field has no orderable values.
        """
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
# together, then sort them by their position in the file. From each tuple, the first field present in all fetched
# documents wins; when no field covers every document, the first present in any of them (see `_pick_order_field`).
# Haystack's document splitters auto-add `split_id` (ordinal) and `split_idx_start` (character offset), some add
# `page_number` and `source_id`; converters add the file name/path.
_GROUP_FIELDS = ("file_name", "file_path", "source_id")
_SORT_FIELDS = ("split_id", "split_idx_start", "page_number")


def _pick_order_field(candidates: tuple[str, ...], documents: list[Document]) -> str | None:
    """
    Pick the first candidate present in all documents, falling back to the first present in any of them.

    Preferring full coverage keeps one document with an odd extra field (e.g. a stray `file_name`) from outranking a
    field every document carries; the any-fallback still orders mixed corpora as well as their partial metadata allows
    (documents missing the field are placed last).

    :param candidates: The candidate field names, in priority order.
    :param documents: The fetched documents.
    :returns: The field to use, or None when no candidate is present in any document.
    """
    fallback = None
    for field in candidates:
        docs_with_field = sum(1 for doc in documents if field in doc.meta)
        if docs_with_field == len(documents) and docs_with_field > 0:
            return field
        if docs_with_field > 0 and fallback is None:
            fallback = field
    return fallback


def _order_documents_for_reading(documents: list[Document]) -> list[Document]:
    """
    Put filter-fetched documents into reading order.

    Filter-fetched documents carry no relevance order, so they are grouped by their parent file (a `_GROUP_FIELDS`
    field, via `MetaFieldGroupingRanker`) and sorted by their position within it (a `_SORT_FIELDS` field), each
    selected by `_pick_order_field`.

    :param documents: The fetched documents.
    :returns: The documents, grouped and sorted when the meta fields allow it.
    """
    group_by = _pick_order_field(_GROUP_FIELDS, documents)
    sort_by = _pick_order_field(_SORT_FIELDS, documents)
    # Both ordering paths sort documents by the raw values of `sort_by`, which raises a TypeError when the field
    # holds mixed, non-comparable types (e.g. int in one document, str in another) — the ranker sorts internally
    # via `list.sort` with the same kind of key. Suppress it and keep the original order then.
    if group_by:
        with contextlib.suppress(TypeError):
            ranker = MetaFieldGroupingRanker(group_by=group_by, sort_docs_by=sort_by)
            return ranker.run(documents=documents)["documents"]
    elif sort_by:
        # Sort by the field value, placing documents with a missing value last.
        with contextlib.suppress(TypeError):
            return sorted(documents, key=lambda d: (sort_by not in d.meta, d.meta.get(sort_by)))
    return documents


def _format_fetch_result(result: dict[str, Any]) -> str:
    """
    Format a fetch-by-filter result: the documents plus paging guidance when more matched than were shown.

    :param result: The result from `FetchDocumentsByFilterTool._fetch_documents`, containing `documents`,
        `total_matched` and `offset`.
    :returns: The formatted documents, or a message nudging the agent to adjust the filter or page onwards.
    """
    documents: list[Document] = result["documents"]
    total_matched: int = result["total_matched"]
    offset: int = result["offset"]
    if not documents:
        if total_matched > 0:
            return f"No documents at offset {offset}: only {total_matched} documents match the filter."
        return "No documents matched the filter. Verify the field names and values with the metadata tools."
    formatted = _format_retrieved_documents(documents)
    remaining = total_matched - offset - len(documents)
    if remaining > 0:
        formatted += (
            f"\n… and {remaining} more documents matched; call again with offset={offset + len(documents)} "
            f"to continue, or narrow the filter."
        )
    return formatted


def _fetch_documents_tool_params(max_docs: int) -> dict[str, Any]:
    """
    Build the input schema for the fetch-by-filter tool: a (required) filter, an optional doc cap and a page offset.

    :param max_docs: The configured ceiling for `max_docs`, embedded in the schema.
    :returns: The JSON schema.
    """
    return {
        "type": "object",
        "properties": {
            "filters": {"type": "object", "description": prompts.FETCH_FILTER_GRAMMAR, "additionalProperties": True},
            "max_docs": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_docs,
                "description": f"How many documents to return at most. Defaults to the maximum, {max_docs}.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "How many matching documents to skip before returning results. "
                    "Use it to page through match sets larger than max_docs. Defaults to 0."
                ),
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
    documents carry. Match sets larger than `max_docs` are paged: each call returns one page plus the total match
    count, and the tool's `offset` input continues where the previous page ended.
    """

    def __init__(self, document_store: DocumentStore, max_docs: int = 10, max_fetch_factor: int = 10) -> None:
        """
        Create the tool.

        :param document_store: The document store to fetch documents from.
        :param max_docs: Ceiling on the number of documents shown to the agent per fetch. Unlike scored retrieval, a
            filter fetch is not bounded by a retriever's `top_k`, so this caps the tool result instead. The LLM can
            request fewer via the tool's optional `max_docs` input, but never more.
        :param max_fetch_factor: How many times the `max_docs` ceiling a filter may match before the fetch is
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
            # We purposefully return a dict with `documents`, `total_matched` and `offset`
            function=self._fetch_documents,
            # This converts the dict into a single string and expects all three keys to be present
            outputs_to_string={"handler": _format_fetch_result},
            # This accumulates the documents into the agent's state
            outputs_to_state={"documents": {"source": "documents", "handler": _accumulate_documents}},
        )

    def _fetch_documents(self, filters: dict[str, Any], max_docs: int | None = None, offset: int = 0) -> dict[str, Any]:
        """
        Fetch the documents matching a metadata filter, in reading order.

        :param filters: The metadata filter, in Haystack filter syntax.
        :param max_docs: How many documents to return at most; clamped to the configured ceiling (`self.max_docs`),
            which is also the default.
        :param offset: How many matching documents to skip before returning results, for paging through match sets
            larger than `max_docs`. Pages line up across calls when the documents carry the reading-order metadata
            fields or the store returns matches in a consistent order.
        :returns: One page of the ordered documents, the page's offset, plus how many documents matched the filter
            overall. Formatted for the LLM by `_format_fetch_result`.
        :raises ValueError: If the filter matches far more documents than can be shown (see below). The `Agent`
            surfaces this to the LLM as an error tool message, so it can recover by narrowing the filter.
        """
        effective_max = max(1, min(max_docs, self.max_docs)) if max_docs is not None else self.max_docs
        offset = max(0, offset)
        # The threshold is based on the configured `self.max_docs` ceiling, not the per-call `effective_max`: it
        # bounds the cost of fetching the whole match set, which is the same however few documents the LLM asked
        # to see — requesting a smaller page must not make the fetch more likely to be refused.
        fetch_limit = self.max_fetch_factor * self.max_docs
        counter = getattr(self.document_store, "count_documents_by_filter", None)
        if counter is not None:
            count = counter(filters=filters)
            if count == 0:
                return {"documents": [], "total_matched": 0, "offset": offset}
            # `filter_documents` has no limit parameter, so on a broad filter a db could return thousands of docs only
            # for us to show `max_docs` of them. When the store supports `count_documents_by_filter`, refuse over-broad
            # filters without fetching anything.
            if count > fetch_limit:
                msg = (
                    f"Filter matches {count} documents, but at most {self.max_docs} can be shown per fetch — retrieving"
                    f" that many would be slow and the result would be mostly truncated anyway, so nothing was fetched."
                    f" This tool is meant for grabbing a small, specific set of documents. Add more conditions to "
                    f"narrow the filter, or use a relevance-based search to get only the most relevant documents."
                )
                raise ValueError(msg)
        documents = _order_documents_for_reading(self.document_store.filter_documents(filters=filters))
        return {
            "documents": documents[offset : offset + effective_max],
            "total_matched": len(documents),
            "offset": offset,
        }

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
