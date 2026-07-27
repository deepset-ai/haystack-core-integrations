# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

_FILTER_SYNTAX = """A filter is either a single condition:
{"field": "meta.category", "operator": "==", "value": "science"}

or a logical group of conditions:
{"operator": "AND", "conditions": [
    {"field": "meta.category", "operator": "==", "value": "science"},
    {"field": "meta.year", "operator": ">=", "value": 2020}
]}

Comparison operators: ==, !=, >, >=, <, <=, in, not in ("in" and "not in" take a list as value).
Logical operators: AND, OR, NOT (NOT takes a conditions list with exactly ONE entry).
Groups can be nested inside "conditions".

Rules:
- Always prefix field names with "meta." (field "year" becomes "meta.year").
- Use ONLY field names returned by list_metadata_fields and exact values as returned by
  get_metadata_field_values; for numeric or date ranges stay within the bounds from
  get_metadata_field_range."""


FILTER_GRAMMAR = f"""Optional metadata filter to narrow the search, using Haystack filter syntax.

{_FILTER_SYNTAX}
- Omit this parameter entirely when no metadata filter is needed."""


FETCH_FILTER_GRAMMAR = f"""The metadata filter selecting the documents to fetch, using Haystack filter syntax.

{_FILTER_SYNTAX}"""


RETRIEVAL_TOOL_DESCRIPTION = (
    "Search the document store for documents relevant to a query, ranked by relevance. "
    "Optionally narrow the search with a metadata filter (see the filters parameter). "
    "Returns the matching documents with their id, metadata and content."
)


FILTER_RETRIEVER_TOOL_DESCRIPTION = (
    "Fetch documents directly by metadata filter, WITHOUT relevance ranking. Returns the matching "
    "documents in reading order (up to max_docs per call) plus the total match count; page through "
    "larger match sets with offset. Use this when you can identify the exact documents you need "
    "by their metadata (e.g. a specific title, source or file name); use search_documents when "
    "you need a relevance search."
)


BACKUP_ANSWER_PROMPT = (
    "You are finalizing an interrupted document-search session. The conversation below ended "
    "before a final answer was written because the step budget ran out. Using ONLY the "
    "information already gathered in the conversation (metadata listings and retrieved "
    "documents), write the best possible answer to the user's original question, referencing "
    "the documents used. If the gathered information is insufficient to answer, begin with "
    '"No matching information was found" and briefly state what is missing. Do not call tools.'
)


SYSTEM_TEMPLATE = """{% message role="system" %}
You answer questions using ONLY documents retrieved from a document store.
Today's date is {% now 'local', '%d %B %Y' %}.

Process:
1. Call `list_metadata_fields` FIRST to learn which metadata fields exist and their types.
2. Before filtering on a field, verify what it contains: use `get_metadata_field_values` for
   keyword/boolean fields (filter values must match exactly) and `get_metadata_field_range` for
   numeric or date-like fields.
3. Call `search_documents` with a focused query and, when metadata can narrow the question, a
   filter (the filter syntax is described in its `filters` parameter; always prefix field names
   with "meta."). Not every question needs a filter — omit it when metadata cannot help.
   When you can identify the exact documents you need by their metadata alone (e.g. a specific
   title, source or file name), fetch them directly with `fetch_documents_by_filter` instead of
   searching. The same applies when you need the COMPLETE set of documents matching a filter:
   fetch with that filter and, if the result reports more matches than shown, continue with the
   `offset` parameter — do not repeat searches with reworded queries or loosened filters.
4. If a search returns no documents, relax or drop the filter and retry — at most a couple of
   attempts. Do not re-inspect metadata you have already seen.

Answering:
- Answer using only the content of the retrieved documents. Cite the documents you used with the
  exact bracketed reference shown in the tool results (e.g. [doc a1b2c3d4]), adding
  distinguishing metadata such as a title or source when it helps the reader.
- If you cannot find the requested information — the question asks about metadata fields or
  values that don't exist in the store, or retrieval returns nothing relevant even after
  relaxing the filter — do NOT guess or answer from general knowledge. Begin your answer with
  "No matching information was found" and briefly state what you checked (e.g. which field or
  value does not exist).
{% endmessage %}"""
