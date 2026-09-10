from haystack_integrations.evaluation import (
    DEFAULT_TOOL_BUDGET,
    ToolRunStats,
    budgets_exceeded,
    resolve_tool_budgets,
)

RETRIEVAL = ("search_documents", "fetch_documents_by_filter")


def test_a_named_group_keeps_one_shared_allowance():
    budgets = resolve_tool_budgets(budgets={RETRIEVAL: 12}, tool_names=list(RETRIEVAL))

    assert budgets == {("fetch_documents_by_filter", "search_documents"): 12}


def test_a_tool_the_eval_case_never_named_is_still_capped():
    """A candidate can gain a tool after the eval case was written; unbudgeted is the one thing it must not be."""
    budgets = resolve_tool_budgets(budgets={RETRIEVAL: 12}, tool_names=[*RETRIEVAL, "web_search"])

    assert budgets[("web_search",)] == DEFAULT_TOOL_BUDGET


def test_the_wildcard_sets_the_allowance_for_everything_left_over():
    budgets = resolve_tool_budgets(budgets={RETRIEVAL: 12, "*": 2}, tool_names=[*RETRIEVAL, "web_search", "finish"])

    assert budgets[("web_search",)] == 2
    assert budgets[("finish",)] == 2
    # Named groups win over the wildcard rather than being narrowed by it.
    assert budgets[("fetch_documents_by_filter", "search_documents")] == 12


def test_a_group_the_agent_does_not_have_is_kept_so_its_allowance_still_reports():
    budgets = resolve_tool_budgets(budgets={"retired_tool": 1}, tool_names=["search_documents"])

    assert budgets == {("retired_tool",): 1, ("search_documents",): DEFAULT_TOOL_BUDGET}


def test_only_the_groups_that_went_over_are_reported():
    stats = ToolRunStats(calls=[("search_documents", {}), ("search_documents", {}), ("list_metadata_fields", {})])
    budgets = {("search_documents",): 1, ("list_metadata_fields",): 5}

    assert budgets_exceeded(stats=stats, budgets=budgets) == {("search_documents",): (2, 1)}


def test_a_group_shares_its_allowance_rather_than_each_tool_having_it():
    """Two calls to two tools of one group spend the group's allowance twice, not once each."""
    stats = ToolRunStats(calls=[("search_documents", {}), ("fetch_documents_by_filter", {})])
    group = ("fetch_documents_by_filter", "search_documents")

    assert budgets_exceeded(stats=stats, budgets={group: 1}) == {group: (2, 1)}
    assert budgets_exceeded(stats=stats, budgets={group: 2}) == {}
