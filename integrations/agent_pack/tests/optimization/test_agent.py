import json

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.evaluation import ModelPrice, RetrievalHarnessEvaluator
from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.optimization import (
    CandidateOutcome,
    ConfigurationDraft,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    propose_candidate,
)
from haystack_integrations.agent_pack.optimization.agent import (
    _describe_environment,
    _render_outcomes,
    _summarized_metrics,
)
from haystack_integrations.agent_pack.optimization.utils import (
    _configuration_id,
    content_digest,
    dump_pipeline,
    load_pipeline,
)

from .test_tools import AcceptEvaluator, agent_yaml, model_yaml, turn_start


def outcome(passed, failures):
    return {
        "question": "q",
        "passed": passed,
        "failures": failures,
        "recall": 1.0,
        "agent_run_digest": {"tool_steps": []},
    }


def latest_revision(messages, yaml):
    """The revision the last successful edit returned, or the starting YAML's when nothing was edited yet."""
    for message in reversed(messages):
        result = message.tool_call_result
        if result is not None and result.origin.tool_name == "edit_config" and not result.error:
            return result.result
    return content_digest(payload=yaml)


def optimizer(llm, **kwargs):
    return create_harness_optimizer_agent(evaluator=AcceptEvaluator(), llm=llm, **kwargs)


def measured(quality, all_tokens_reported=True, details=None, **fields):
    """A measurement from a stub harness, which reports its quality under the key the objectives name."""
    return EvalMetrics(
        all_tokens_reported=all_tokens_reported, details={"quality": quality, **(details or {})}, **fields
    )


class TestCreateOptimizerAgent:
    def test_defaults_and_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(evaluator=AcceptEvaluator(), additional_instructions="Keep the corpus.")
        assert isinstance(agent.chat_generator, OpenAIResponsesChatGenerator)
        # Pinned so that changing what the search itself costs stays a deliberate decision.
        assert agent.chat_generator.model == "gpt-5.6-terra"
        assert _describe_environment() in agent.system_prompt
        assert agent.system_prompt.endswith("Keep the corpus.")
        assert agent.chat_generator.generation_kwargs["reasoning"] == {"effort": "low"}

    def test_complete_agent(self, monkeypatch):
        """
        Every tool the exit conditions name is the agent's own, along with the state those tools share. The system
        prompt refers to the tools by these names.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(evaluator=AcceptEvaluator())
        assert [tool.name for tool in agent.tools] == [
            "read_config",
            "edit_config",
            "validate_config",
            "submit_candidate",
            "restore_candidate",
            "finish",
            "inspect_component",
        ]
        assert agent.exit_conditions == ["submit_candidate", "finish"]
        assert {"draft", "known", "submitted", "finish_reason", "validation_failures"} <= set(agent.state_schema)

    def test_editing_tools_run_in_call_order(self):
        """
        Called without `propose_candidate`, the agent edits the draft it is given. An edit and a validation requested in
        one step run in that order, so the validation sees the edit.
        """
        draft, known = turn_start()
        revision = draft.revision
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
                [ToolCall("finish", {"reason": "nothing worth measuring"}, id="finish")],
            ]
        )
        agent = optimizer(
            llm=MockChatGenerator(response_fn=lambda _messages: ChatMessage.from_assistant(tool_calls=next(steps)))
        )
        result = agent.run(messages=[ChatMessage.from_user("Propose the next candidate.")], draft=draft, known=known)
        assert "model: cheap" in result["draft"].yaml
        # The validation ran after the edit, so it validated the edited revision
        assert result["draft"].validated_revision == result["draft"].revision
        assert result["finish_reason"] == "nothing worth measuring"

    def test_serialization_roundtrip(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(evaluator=RetrievalHarnessEvaluator(k=3), loader=load_pipeline)
        restored = Agent.from_dict(agent.to_dict())
        assert [tool.name for tool in restored.tools] == [tool.name for tool in agent.tools]
        assert restored.exit_conditions == ["submit_candidate", "finish"]
        validate = restored.tools[2]
        assert (validate.evaluator.k, validate.loader) == (3, load_pipeline)
        assert restored.state_schema["draft"]["type"] is ConfigurationDraft

    def test_documentation_tools(self, monkeypatch):
        mcp = pytest.importorskip("haystack_integrations.tools.mcp")
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = create_harness_optimizer_agent(evaluator=AcceptEvaluator(), documentation_tools=True)
        assert agent.tools[-2].name == "inspect_component"
        assert isinstance(agent.tools[-1], mcp.MCPToolset)


class TestProposeCandidate:
    def test_validation_gates_the_submission(self):
        """A draft that breaks validation, or was never validated, is refused until it is repaired and validated."""
        reference = Agent(chat_generator=MockChatGenerator(model="reference"))
        reference_yaml = agent_yaml(reference)
        steps = iter(
            [
                ("edit_config", "model: reference", "model: [broken"),
                ("validate_config",),
                ("edit_config", "model: [broken", "model: cheap"),
                ("submit_candidate",),
                ("validate_config",),
                ("submit_candidate",),
            ]
        )

        def respond(messages):
            name, *edit = next(steps)
            revision = latest_revision(messages=messages, yaml=reference_yaml)
            arguments = {
                "edit_config": lambda: {"old": edit[0], "new": edit[1], "expected_revision": revision},
                "validate_config": dict,
                "submit_candidate": lambda: {"expected_revision": revision, "rationale": "reduce cost"},
            }[name]()
            return ChatMessage.from_assistant(tool_calls=[ToolCall(name, arguments, id=name)])

        result = propose_candidate(
            optimizer_agent=optimizer(llm=MockChatGenerator(response_fn=respond)),
            reference=reference,
            reference_yaml=reference_yaml,
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert "model: cheap" in result.candidate.yaml
        assert len(result.validation_failures) == 1

    def test_reports_remaining_measurements(self):
        """A submission costs a full pass over the evaluation set, so the budget has to be visible to economize."""
        seen = []

        def respond(messages):
            seen.append([message.text or "" for message in messages])
            return ChatMessage.from_assistant("done")

        propose_candidate(
            optimizer_agent=optimizer(llm=MockChatGenerator(response_fn=respond), max_agent_steps=1),
            reference_yaml=agent_yaml(),
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
                optimizer_agent=optimizer(llm=MockChatGenerator(response_fn=respond), max_agent_steps=1),
                reference_yaml=agent_yaml(),
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

    def test_starts_from_the_base(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        seen = []

        def respond(messages):
            seen.append(messages[-1].text or "")
            return ChatMessage.from_assistant("done")

        propose_candidate(
            optimizer_agent=optimizer(llm=MockChatGenerator(response_fn=respond), max_agent_steps=1),
            reference=Agent(chat_generator=MockChatGenerator()),
            reference_yaml=agent_yaml(),
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
            candidates={cheap_id: cheap},
            base_id=cheap_id,
        )
        assert f"edited from `{cheap_id}`" in seen[0]
        assert "model: cheap" in seen[0]

    def test_unknown_base(self):
        with pytest.raises(ValueError, match="Unknown base"):
            propose_candidate(
                optimizer_agent=optimizer(llm=MockChatGenerator("done")),
                reference=Agent(chat_generator=MockChatGenerator()),
                reference_yaml=agent_yaml(),
                prices={},
                objectives=OptimizationObjectives(quality_metric="quality"),
                baseline=measured(quality=1, durations=[1]),
                history=[],
                base_id="never-measured",
            )

    def test_plain_text_does_not_submit(self):
        result = propose_candidate(
            optimizer_agent=optimizer(llm=MockChatGenerator("done"), max_agent_steps=2),
            reference_yaml=agent_yaml(),
            reference=Agent(chat_generator=MockChatGenerator()),
            prices={},
            objectives=OptimizationObjectives(quality_metric="quality"),
            baseline=measured(quality=1, durations=[1]),
            history=[],
        )
        assert result.candidate is None
        assert result.finish_reason is None

    def test_reference_without_tools(self):
        """A Pipeline that is not an Agent has no tools, and a heading over an empty list only costs cached prefix."""
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore()))
        seen = []

        def respond(messages):
            # messages[0] is the system prompt; the stable context is the first user message.
            seen.append(messages[1].text or "")
            return ChatMessage.from_assistant("done")

        propose_candidate(
            optimizer_agent=create_harness_optimizer_agent(
                evaluator=AcceptEvaluator(),
                llm=MockChatGenerator(response_fn=respond),
                max_agent_steps=1,
                loader=load_pipeline,
            ),
            reference=pipeline,
            reference_yaml=dump_pipeline(pipeline),
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
            eval_cases=[outcome(True, []), outcome(False, ["a", "b"]), outcome(False, ["a"])],
            details={"model": "m"},
        )
        summarized = _summarized_metrics(metrics=metrics)
        # Each failure is counted by how many eval cases it appeared in
        assert summarized["eval_case_summary"] == {"total": 3, "passed": 1, "failures": {"a": 2, "b": 1}}
        assert "eval_cases" not in summarized
        # Everything that is not the listing survives untouched.
        assert summarized["details"] == {"quality": 0.5, "model": "m"}

    def test_summarized_metrics_without_eval_cases(self):
        metrics = measured(quality=0.5, durations=[1])
        assert _summarized_metrics(metrics=metrics) == metrics.to_dict()

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
