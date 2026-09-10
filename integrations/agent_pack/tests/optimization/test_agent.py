import json

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.optimization import (
    ConfigurationWorkspace,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    dump_pipeline,
    load_pipeline,
    propose_candidate,
)
from haystack_integrations.agent_pack.optimization.agent import (
    _create_haystack_documentation_mcp_toolset,
    _describe_environment,
    _documentation_result,
    _summarize_eval_case_details,
)
from haystack_integrations.evaluation.dataclasses import EvaluationMetrics

from .test_workspace import agent_yaml


def test_defaults_and_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    agent = create_harness_optimizer_agent(additional_instructions="Keep the corpus.")
    assert isinstance(agent.chat_generator, OpenAIResponsesChatGenerator)
    # Pinned so that changing what the search itself costs stays a deliberate decision.
    assert agent.chat_generator.model == "gpt-5.6-luna"
    assert _describe_environment() in agent.system_prompt
    assert agent.system_prompt.endswith("Keep the corpus.")
    assert agent.chat_generator.generation_kwargs["reasoning"] == {"effort": "low"}


def test_documentation_toolset_is_optional_and_read_only():
    pytest.importorskip("haystack_integrations.tools.mcp")
    docs = _create_haystack_documentation_mcp_toolset()
    assert docs.tool_names == ["search_haystack_docs"]
    assert docs.eager_connect is False


def docs_payload(**body):
    """The shape the documentation server answers with: a payload serialized inside an MCP envelope."""
    return json.dumps({"meta": None, "content": [{"type": "text", "text": json.dumps(body)}]})


def test_documentation_search_reports_documentation_not_the_servers_debug_output():
    """Measured against the live server, the debug payload is 94% of the answer and says nothing about Haystack."""
    payload = docs_payload(
        documents=[{"content": "LLMRanker reorders documents.", "meta": {"url": "https://docs/llmranker"}}],
        _debug={"pipeline": "x" * 5000},
    )

    result = _documentation_result(payload)

    assert result == "[https://docs/llmranker]\nLLMRanker reorders documents."
    assert "_debug" not in result


def test_a_long_documentation_section_is_capped():
    payload = docs_payload(documents=[{"content": "x" * 10_000, "meta": {}}])

    assert len(_documentation_result(payload)) <= 4100


def test_an_unexpected_answer_is_passed_through_rather_than_swallowed():
    """A server that changes shape must not silently look like an empty search."""
    assert _documentation_result("not json at all") == "not json at all"
    assert _documentation_result(docs_payload(unexpected=1)).startswith('{"meta"')


def test_no_matching_documentation_says_so():
    assert _documentation_result(docs_payload(documents=[])) == "No documentation matched."


def test_optimizer_repairs_yaml_before_submitting(tmp_path):
    reference = Agent(chat_generator=MockChatGenerator(model="reference"))
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml(reference))
    stage = iter(["break", "validate", "repair", "validate", "submit"])

    def respond(_messages):
        current = workspace.read_config()
        action = next(stage)
        if action in ("break", "repair"):
            old, new = (
                ("model: reference", "model: [broken") if action == "break" else ("model: [broken", "model: cheap")
            )
            call = ToolCall(
                "edit_config", {"old": old, "new": new, "expected_revision": current["revision"]}, id=action
            )
        elif action == "validate":
            call = ToolCall("validate_config", {}, id=action)
        else:
            call = ToolCall(
                "submit_candidate", {"expected_revision": current["revision"], "rationale": "reduce cost"}, id=action
            )
        return ChatMessage.from_assistant(tool_calls=[call])

    result = propose_candidate(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond)),
        workspace=workspace,
        reference=reference,
        pricing=ModelPriceCatalog([]),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1, cost=1, latency_ms=1),
        history=[],
    )
    assert result is not None
    assert "model: cheap" in result.yaml
    assert len(workspace.validation_failures) == 1


def test_the_optimizer_is_told_how_many_measurements_remain(tmp_path):
    """A submission costs a full pass over the evaluation set, so the budget has to be visible to economize."""
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    seen = []

    def respond(messages):
        seen.append([message.text or "" for message in messages])
        return ChatMessage.from_assistant("done")

    propose_candidate(
        optimizer_agent=create_harness_optimizer_agent(
            chat_generator=MockChatGenerator(response_fn=respond), max_agent_steps=1
        ),
        workspace=workspace,
        reference=Agent(chat_generator=MockChatGenerator()),
        pricing=ModelPriceCatalog([]),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1, cost=1, latency_ms=1),
        history=[],
        remaining_evaluations=3,
    )

    stable, _outcomes, volatile = seen[0][1], seen[0][2], seen[0][3]
    # The budget changes every turn, so it belongs in the volatile message and not in the cacheable prefix.
    assert "3 evaluations remain" in volatile
    assert "3 evaluations remain" not in stable
    assert "## Objectives" in stable


def test_only_the_most_recently_measured_configuration_is_described_eval_case_by_eval_case(tmp_path):
    """On the first turn that is the reference; once a candidate has been measured, the reference is summarized."""
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    baseline = EvaluationMetrics(
        quality=0.5,
        latency_ms=1,
        details={
            "model": "reference",
            "eval_cases": [{"passed": True, "failures": []}, {"passed": False, "failures": ["r"]}],
        },
    )
    seen = []

    def respond(messages):
        seen.append(json.loads(messages[1].text.split("## Reference measurement")[1].strip()))
        return ChatMessage.from_assistant("done")

    def propose(history):
        propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(
                chat_generator=MockChatGenerator(response_fn=respond), max_agent_steps=1
            ),
            workspace=workspace,
            reference=Agent(chat_generator=MockChatGenerator()),
            pricing=ModelPriceCatalog([]),
            objectives=OptimizationObjectives(),
            baseline=baseline,
            history=history,
        )

    propose(history=[])
    propose(history=[{"candidate_id": "c1", "metrics": None}])

    first, later = seen
    assert [eval_case["passed"] for eval_case in first["details"]["eval_cases"]] == [True, False]
    assert "eval_cases" not in later["details"]
    assert later["details"]["eval_case_summary"] == {"total": 2, "passed": 1, "failures": {"r": 1}}
    # The measurement itself is untouched; only what the request carries changes.
    assert baseline.details["eval_cases"][0]["passed"] is True


def test_plain_text_does_not_submit_or_run_forever(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    result = propose_candidate(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator("done"), max_agent_steps=2),
        workspace=workspace,
        reference=Agent(chat_generator=MockChatGenerator()),
        pricing=ModelPriceCatalog([]),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1, cost=1, latency_ms=1),
        history=[],
    )
    assert result is None
    assert workspace.submitted is None


def test_failed_submission_keeps_the_agent_running_for_repair(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    current = workspace.read_config()
    workspace.edit_config("model: reference", "model: cheap", current["revision"])
    steps = iter(["submit_candidate", "validate_config", "submit_candidate"])

    def respond(_messages):
        name = next(steps)
        arguments = (
            {}
            if name == "validate_config"
            else {
                "expected_revision": workspace.read_config()["revision"],
                "rationale": "test validation gate",
            }
        )
        return ChatMessage.from_assistant(tool_calls=[ToolCall(name, arguments, id=name)])

    result = propose_candidate(
        optimizer_agent=create_harness_optimizer_agent(chat_generator=MockChatGenerator(response_fn=respond)),
        workspace=workspace,
        reference=Agent(chat_generator=MockChatGenerator()),
        pricing=ModelPriceCatalog([]),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1, cost=1, latency_ms=1),
        history=[],
    )
    assert result is not None


def test_a_reference_without_tools_gets_no_tools_section(tmp_path):
    """A Pipeline that is not an Agent has no tools, and a heading over an empty list only costs cached prefix."""
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore()))
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", dump_pipeline(pipeline), loader=load_pipeline)
    seen = []

    def respond(messages):
        # messages[0] is the system prompt; the stable context is the first user message.
        seen.append(messages[1].text or "")
        return ChatMessage.from_assistant("done")

    propose_candidate(
        optimizer_agent=create_harness_optimizer_agent(
            chat_generator=MockChatGenerator(response_fn=respond), max_agent_steps=1
        ),
        workspace=workspace,
        reference=pipeline,
        pricing=ModelPriceCatalog([]),
        objectives=OptimizationObjectives(),
        baseline=EvaluationMetrics(quality=1, cost=1, latency_ms=1),
        history=[],
    )

    assert "## Tools available to the reference" not in seen[0]
    assert "## Objectives" in seen[0]


def outcome(passed, failures):
    return {
        "question": "q",
        "passed": passed,
        "failures": failures,
        "recall": 1.0,
        "agent_run_digest": {"tool_steps": []},
    }


def test_old_eval_case_listings_become_a_count_of_how_they_ended():
    history = [
        {"metrics": {"details": {"model": "m", "eval_cases": [outcome(True, []), outcome(False, ["recall_below_1"])]}}}
    ]

    summarized = _summarize_eval_case_details(payload=history)

    assert summarized[0]["metrics"]["details"]["eval_case_summary"] == {
        "total": 2,
        "passed": 1,
        "failures": {"recall_below_1": 1},
    }
    # Everything that is not the listing survives untouched.
    assert summarized[0]["metrics"]["details"]["model"] == "m"
    assert "eval_cases" not in summarized[0]["metrics"]["details"]


def test_summarizing_is_safe_on_a_failed_candidate_with_no_metrics():
    history = [{"metrics": None, "failure": "boom"}]

    assert _summarize_eval_case_details(payload=history) == history


def test_a_failure_hit_by_several_eval_cases_is_counted_once_per_eval_case():
    """The summary is what tells an optimizer which failure is worth attacking, so the counts have to add up."""
    history = [{"metrics": {"details": {"eval_cases": [outcome(False, ["a", "b"]), outcome(False, ["a"])]}}}]

    summarized = _summarize_eval_case_details(payload=history)

    assert summarized[0]["metrics"]["details"]["eval_case_summary"]["failures"] == {"a": 2, "b": 1}
