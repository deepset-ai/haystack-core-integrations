import json

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.evaluation import ModelPrice
from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.optimization import (
    CandidateOutcome,
    ConfigurationEditorToolset,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    propose_candidate,
)
from haystack_integrations.agent_pack.optimization.agent import (
    _describe_environment,
    _render_outcomes,
    _summarized_metrics,
)
from haystack_integrations.agent_pack.optimization.utils import dump_pipeline, load_pipeline

from .test_tools import agent_yaml


def outcome(passed, failures):
    return {
        "question": "q",
        "passed": passed,
        "failures": failures,
        "recall": 1.0,
        "agent_run_digest": {"tool_steps": []},
    }


def measured(quality, all_tokens_reported=True, details=None, **fields):
    """A measurement from a stub harness, which reports its quality under the key the objectives name."""
    return EvalMetrics(
        all_tokens_reported=all_tokens_reported, details={"quality": quality, **(details or {})}, **fields
    )


class TestCreateOptimizerAgent:
    def test_defaults_and_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(additional_instructions="Keep the corpus.")
        assert isinstance(agent.chat_generator, OpenAIResponsesChatGenerator)
        # Pinned so that changing what the search itself costs stays a deliberate decision.
        assert agent.chat_generator.model == "gpt-5.6-terra"
        assert _describe_environment() in agent.system_prompt
        assert agent.system_prompt.endswith("Keep the corpus.")
        assert agent.chat_generator.generation_kwargs["reasoning"] == {"effort": "low"}

    def test_tools_and_exit_conditions(self, monkeypatch):
        """The editing tools come from the editor passed for each run, so the agent holds only its own."""
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent()
        assert [tool.name for tool in agent.tools] == ["inspect_component"]
        assert agent.exit_conditions == ["submit_candidate", "finish"]

    def test_run_directly(self):
        """Called without `propose_candidate`, the agent edits whatever editor it is given."""
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        calls = iter(
            [
                ToolCall("read_config", {}, id="read"),
                ToolCall("finish", {"reason": "nothing worth measuring"}, id="finish"),
            ]
        )
        agent = create_harness_optimizer_agent(
            llm=MockChatGenerator(response_fn=lambda _messages: ChatMessage.from_assistant(tool_calls=[next(calls)]))
        )
        agent.run(messages=[ChatMessage.from_user("Propose the next candidate.")], tools=[editor, *agent.tools])
        # The tools changed the editor that was passed in, not a copy of it
        assert editor.finished
        assert editor.finish_reason == "nothing worth measuring"

    def test_editing_tools_run_in_call_order(self):
        """An edit and a validation requested in one step run in that order, so the validation sees the edit."""
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        revision = editor._read_config()["revision"]
        steps = iter(
            [
                [
                    ToolCall(
                        "edit_config",
                        {"old": "model: reference", "new": "model: cheap", "expected_revision": revision},
                        id="edit",
                    ),
                    ToolCall("validate_config", {}, id="validate"),
                ],
                [ToolCall("finish", {"reason": "done"}, id="finish")],
            ]
        )
        agent = create_harness_optimizer_agent(
            llm=MockChatGenerator(response_fn=lambda _messages: ChatMessage.from_assistant(tool_calls=next(steps)))
        )
        agent.run(messages=[ChatMessage.from_user("Propose the next candidate.")], tools=[editor, *agent.tools])
        edited = editor._read_config()
        assert "model: cheap" in edited["yaml"]
        # The validation ran after the edit, so it validated the edited revision
        assert editor.validated_revision == edited["revision"]

    def test_serialization_roundtrip(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        restored = Agent.from_dict(create_harness_optimizer_agent().to_dict())
        assert [tool.name for tool in restored.tools][-1] == "inspect_component"
        assert restored.exit_conditions == ["submit_candidate", "finish"]

    def test_documentation_tools(self, monkeypatch):
        mcp = pytest.importorskip("haystack_integrations.tools.mcp")
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(documentation_tools=True)
        assert agent.tools[-2].name == "inspect_component"
        assert isinstance(agent.tools[-1], mcp.MCPToolset)


class TestProposeCandidate:
    def test_repairs_yaml_before_submitting(self):
        reference = Agent(chat_generator=MockChatGenerator(model="reference"))
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml(reference))
        stage = iter(["break", "validate", "repair", "validate", "submit"])

        def respond(_messages):
            current = editor._read_config()
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
                    "submit_candidate",
                    {"expected_revision": current["revision"], "rationale": "reduce cost"},
                    id=action,
                )
            return ChatMessage.from_assistant(tool_calls=[call])

        result = propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(llm=MockChatGenerator(response_fn=respond)),
            editor=editor,
            reference=reference,
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert result is not None
        assert "model: cheap" in result.yaml
        assert len(editor.validation_failures) == 1

    def test_reports_remaining_measurements(self):
        """A submission costs a full pass over the evaluation set, so the budget has to be visible to economize."""
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        seen = []

        def respond(messages):
            seen.append([message.text or "" for message in messages])
            return ChatMessage.from_assistant("done")

        propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(
                llm=MockChatGenerator(response_fn=respond), max_agent_steps=1
            ),
            editor=editor,
            reference=Agent(chat_generator=MockChatGenerator()),
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
            remaining_evaluations=3,
        )
        stable, _outcomes, volatile = seen[0][1], seen[0][2], seen[0][3]
        # The budget changes every turn, so it belongs in the volatile message and not in the cacheable prefix.
        assert "3 evaluations remain" in volatile
        assert "3 evaluations remain" not in stable
        assert "## Objectives" in stable

    def test_only_the_latest_run_is_described_in_full(self):
        """On the first turn that is the reference; once a candidate has been measured, the reference is summarized."""
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        baseline = measured(
            quality=0.5,
            durations=[1],
            eval_cases=[{"passed": True, "failures": []}, {"passed": False, "failures": ["r"]}],
            details={"model": "reference"},
            model_usage={"reference": ModelTokenUsage(input_tokens=1_000_000)},
        )
        seen = []

        def respond(messages):
            seen.append(json.loads(messages[1].text.split("## Reference measurement")[1].strip()))
            return ChatMessage.from_assistant("done")

        def propose(history):
            propose_candidate(
                optimizer_agent=create_harness_optimizer_agent(
                    llm=MockChatGenerator(response_fn=respond), max_agent_steps=1
                ),
                editor=editor,
                reference=Agent(chat_generator=MockChatGenerator()),
                prices={"reference": ModelPrice(input_cost_per_million=0.25, output_cost_per_million=0.0)},
                objectives=OptimizationObjectives(quality_metric="quality"),
                baseline=baseline,
                history=history,
            )

        propose(history=[])
        propose(history=[CandidateOutcome(candidate_id="c1")])
        first, later = seen
        assert first["cost"] == later["cost"] == 0.25
        assert [eval_case["passed"] for eval_case in first["eval_cases"]] == [True, False]
        assert "eval_cases" not in later
        assert later["eval_case_summary"] == {"total": 2, "passed": 1, "failures": {"r": 1}}
        # The measurement itself is untouched; only what the request carries changes.
        assert baseline.eval_cases[0]["passed"] is True

    def test_plain_text_does_not_submit(self):
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        result = propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(llm=MockChatGenerator("done"), max_agent_steps=2),
            editor=editor,
            reference=Agent(chat_generator=MockChatGenerator()),
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert result is None
        assert editor.submitted is None

    def test_failed_submission_allows_repair(self):
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        current = editor._read_config()
        editor._edit_config("model: reference", "model: cheap", current["revision"])
        steps = iter(["submit_candidate", "validate_config", "submit_candidate"])

        def respond(_messages):
            name = next(steps)
            arguments = (
                {}
                if name == "validate_config"
                else {
                    "expected_revision": editor._read_config()["revision"],
                    "rationale": "test validation gate",
                }
            )
            return ChatMessage.from_assistant(tool_calls=[ToolCall(name, arguments, id=name)])

        result = propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(llm=MockChatGenerator(response_fn=respond)),
            editor=editor,
            reference=Agent(chat_generator=MockChatGenerator()),
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert result is not None

    def test_reference_without_tools(self):
        """A Pipeline that is not an Agent has no tools, and a heading over an empty list only costs cached prefix."""
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore()))
        editor = ConfigurationEditorToolset(reference_yaml=dump_pipeline(pipeline), loader=load_pipeline)
        seen = []

        def respond(messages):
            # messages[0] is the system prompt; the stable context is the first user message.
            seen.append(messages[1].text or "")
            return ChatMessage.from_assistant("done")

        propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(
                llm=MockChatGenerator(response_fn=respond), max_agent_steps=1
            ),
            editor=editor,
            reference=pipeline,
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert "## Tools available to the reference" not in seen[0]
        assert "## Objectives" in seen[0]


class TestPromptSummaries:
    def test_summarized_metrics_replaces_the_listing_with_counts(self):
        metrics = measured(
            quality=0.5,
            durations=[1],
            eval_cases=[outcome(True, []), outcome(False, ["recall_below_1"])],
            details={"model": "m"},
        )
        summarized = _summarized_metrics(metrics=metrics)
        assert summarized["eval_case_summary"] == {"total": 2, "passed": 1, "failures": {"recall_below_1": 1}}
        assert "eval_cases" not in summarized
        # Everything that is not the listing survives untouched.
        assert summarized["details"] == {"quality": 0.5, "model": "m"}

    def test_summarized_metrics_without_eval_cases(self):
        metrics = measured(quality=0.5, durations=[1])
        assert _summarized_metrics(metrics=metrics) == metrics.to_dict()

    def test_counts_a_failure_once_per_eval_case(self):
        """The summary is what tells an optimizer which failure is worth attacking, so the counts have to add up."""
        metrics = measured(quality=0.5, durations=[1], eval_cases=[outcome(False, ["a", "b"]), outcome(False, ["a"])])
        assert _summarized_metrics(metrics=metrics)["eval_case_summary"]["failures"] == {"a": 2, "b": 1}

    def test_render_outcomes(self):
        history = [
            CandidateOutcome(
                candidate_id="c1",
                rationale="raise top_k",
                metrics=measured(
                    quality=0.5, durations=[1], eval_cases=[outcome(True, []), outcome(False, ["recall_below_1"])]
                ),
                cost=0.25,
                gate_failures=("usage_incomplete",),
            ),
            CandidateOutcome(candidate_id="c2", parent_id="c1", failure="boom"),
        ]
        rendered = _render_outcomes(history=history)
        assert "### 1. `c1` from `reference` — gates usage_incomplete" in rendered
        assert "cost $0.2500 | quality 0.500 | 1/2 eval cases clean | failures recall_below_1 x1" in rendered
        assert "hypothesis: raise top_k" in rendered
        assert "### 2. `c2` from `c1` — gates passed" in rendered
        assert "not measured" in rendered
        assert "failed to evaluate: boom" in rendered
