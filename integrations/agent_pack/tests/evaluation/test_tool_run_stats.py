from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.agent_pack.evaluation import ToolRunStats, extract_tool_run_stats


def call_and_result(name, arguments=None, result="payload", error=False):
    """One assistant turn calling a tool, followed by the tool's answer."""
    call = ToolCall(name, arguments or {}, id=name)
    return [ChatMessage.from_assistant(tool_calls=[call]), ChatMessage.from_tool(result, origin=call, error=error)]


def test_calls_and_errors_are_extracted_in_the_order_they_happened():
    messages = [
        *call_and_result("list_metadata_fields"),
        *call_and_result("search_documents", {"query": "CRISPR"}),
        *call_and_result("get_metadata_field_values", result="field 'nope' does not exist", error=True),
    ]

    stats = extract_tool_run_stats(messages=messages)

    assert [name for name, _ in stats.calls] == [
        "list_metadata_fields",
        "search_documents",
        "get_metadata_field_values",
    ]
    assert stats.calls[1] == ("search_documents", {"query": "CRISPR"})
    # The failing tool names itself, so a report can say which one refused and why.
    assert stats.errors == [("get_metadata_field_values", "field 'nope' does not exist")]


def test_a_group_of_tools_is_counted_as_one_allowance():
    """Harnesses budget retrieval as a whole rather than per tool, so a group counts together."""
    stats = ToolRunStats(calls=[("search_documents", {}), ("fetch_documents_by_filter", {}), ("finish", {})])

    assert stats.calls_to(tools="search_documents") == 1
    assert stats.calls_to(tools=("search_documents", "fetch_documents_by_filter")) == 2
    assert stats.calls_to(tools=["nothing_called"]) == 0


def test_only_calls_that_passed_something_for_the_argument_are_counted():
    stats = ToolRunStats(
        calls=[
            ("search_documents", {"query": "x", "filters": {"field": "meta.year"}}),
            ("search_documents", {"query": "x", "filters": None}),
            ("search_documents", {"query": "x"}),
        ]
    )

    assert stats.calls_to(tools="search_documents") == 3
    assert stats.calls_with_argument(tools="search_documents", argument="filters") == 1


def test_ordering_holds_only_when_the_first_tool_actually_ran_first():
    metadata, retrieval = "list_metadata_fields", ("search_documents", "fetch_documents_by_filter")
    inspected = ToolRunStats(calls=[(metadata, {}), ("search_documents", {})])
    retrieved = ToolRunStats(calls=[("search_documents", {}), (metadata, {})])

    assert inspected.called_before(tools=metadata, other=retrieval) is True
    assert retrieved.called_before(tools=metadata, other=retrieval) is False
    # Neither ran, so nothing came first and the expectation is not met.
    assert ToolRunStats().called_before(tools=metadata, other=retrieval) is False
