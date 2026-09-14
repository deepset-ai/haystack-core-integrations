import json

from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.evaluation.agent_run_digest import (
    AgentRunDigestPolicy,
    digest_agent_run,
)


def agent_result(*, result_text="one document", error=False, answer="the answer", calls=1):
    """Build an Agent.run output with the given number of tool calls."""
    messages = [ChatMessage.from_user("q")]
    for index in range(calls):
        call = ToolCall(tool_name="search_documents", arguments={"query": f"q{index}"}, id=f"c{index}")
        messages.append(ChatMessage.from_assistant(tool_calls=[call]))
        messages.append(ChatMessage.from_tool(result_text, origin=call, error=error))
    messages.append(ChatMessage.from_assistant(answer))
    return {
        "messages": messages,
        "last_message": messages[-1],
        "exit_reason": "text",
        "step_count": calls + 1,
        "tool_call_counts": {"search_documents": calls, "fetch_documents_by_filter": 0},
    }


class TestDigestAgentRun:
    def test_pairs_calls_with_results(self):
        """A call and its result belong together, and never-called tools still have to be discoverable."""
        digest = digest_agent_run(result=agent_result())
        assert digest["tool_steps"] == [
            {"tool": "search_documents", "arguments": '{"query": "q0"}', "result": "one document", "error": False}
        ]
        # `tool_call_counts` names every tool the Agent could call, including the one it never did.
        assert digest["tool_call_counts"]["fetch_documents_by_filter"] == 0
        assert digest["exit_reason"] == "text"
        assert digest["answer_excerpt"] == "the answer"

    def test_reports_tool_errors_verbatim(self):
        """A tool's own explanation of a refusal is the most direct evidence of a misconfiguration."""
        refusal = "Filter matches 59 documents, but at most 2 can be shown per fetch"
        digest = digest_agent_run(result=agent_result(result_text=refusal, error=True))
        assert digest["tool_steps"][0]["error"] is True
        assert digest["tool_steps"][0]["result"] == refusal

    def test_result_without_messages(self):
        """An Agent result that carries no transcript still produces a usable digest."""
        digest = digest_agent_run(result={"exit_reason": "max_agent_steps"})
        assert digest["tool_steps"] == []
        assert digest["answer_excerpt"] == ""
        assert digest["exit_reason"] == "max_agent_steps"

    def test_is_json_serializable(self):
        """The digest goes into a JSON request and a JSON-lines journal."""
        json.dumps(digest_agent_run(result=agent_result()))


class TestAgentRunDigestPolicy:
    def test_truncation_announces_itself(self):
        """A reader that cannot tell a prefix from a complete listing will treat a partial one as exhaustive."""
        policy = AgentRunDigestPolicy(max_result_chars=10, max_answer_chars=6)
        digest = digest_agent_run(result=agent_result(result_text="x" * 30, answer="y" * 20), policy=policy)
        step = digest["tool_steps"][0]
        assert step["result"].startswith("x" * 10)
        assert "20 more characters omitted" in step["result"]
        assert step["result_truncated"] is True
        assert digest["answer_truncated"] is True

    def test_tools_kept_in_full(self):
        """An introspection listing is only usable as evidence when it is known to be complete."""
        policy = AgentRunDigestPolicy(max_result_chars=10, keep_full_results_for=frozenset({"search_documents"}))
        digest = digest_agent_run(result=agent_result(result_text="x" * 30), policy=policy)
        assert digest["tool_steps"][0]["result"] == "x" * 30
        assert "result_truncated" not in digest["tool_steps"][0]

    def test_call_cap(self):
        """A capped trace still has to say that the run was longer than what it shows."""
        digest = digest_agent_run(result=agent_result(calls=5), policy=AgentRunDigestPolicy(max_tool_calls=2))
        assert len(digest["tool_steps"]) == 2
        assert digest["omitted_tool_calls"] == 3
