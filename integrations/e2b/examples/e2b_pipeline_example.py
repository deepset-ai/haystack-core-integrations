# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Haystack Pipeline with an Agent and E2B sandbox tools.

Demonstrates that a Pipeline containing an Agent with E2BToolset can be:
  1. Serialised to YAML
  2. Written to disk
  3. Loaded back from YAML with full sandbox config intact

All four tools (run_bash_command, read_file, write_file, list_directory)
share a single E2BSandbox after the round-trip, so the agent operates
in one live sandbox environment.

Requirements:
    pip install e2b-haystack openai

Environment variables:
    E2B_API_KEY    - your E2B API key
    OPENAI_API_KEY - your OpenAI API key
"""

import tempfile
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.e2b import E2BToolset


def build_pipeline() -> Pipeline:
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=E2BToolset(sandbox_template="base", timeout=120),
        system_prompt=(
            "You are a helpful coding assistant with access to a live Linux sandbox. "
            "Use the available tools freely to explore, write files, and run commands."
        ),
        max_agent_steps=10,
    )
    pipeline = Pipeline()
    pipeline.add_component("agent", agent)
    return pipeline


def roundtrip_yaml(pipeline: Pipeline) -> Pipeline:
    """Serialise to YAML, save to a temp file, load it back."""
    yaml_str = pipeline.dumps()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        yaml_path = Path(f.name)

    print(f"Pipeline YAML written to {yaml_path}\n")
    print(yaml_str)
    print("---\n")

    return Pipeline.loads(yaml_path.read_text())


def verify_roundtrip(original: Pipeline, restored: Pipeline) -> None:
    """Check that the restored pipeline has the same structure."""
    orig_agent: Agent = original.get_component("agent")
    rest_agent: Agent = restored.get_component("agent")

    orig_ts: E2BToolset = orig_agent.tools  # type: ignore[assignment]
    rest_ts: E2BToolset = rest_agent.tools  # type: ignore[assignment]

    assert type(rest_ts).__name__ == "E2BToolset", "Toolset type mismatch"
    assert [t.name for t in rest_ts] == [t.name for t in orig_ts], "Tool names mismatch"
    assert rest_ts.sandbox.sandbox_template == orig_ts.sandbox.sandbox_template
    assert rest_ts.sandbox.timeout == orig_ts.sandbox.timeout

    sandbox_ids = {id(t._e2b_sandbox) for t in rest_ts}
    assert len(sandbox_ids) == 1, "Tools should share a single sandbox after round-trip"

    print("All assertions passed: YAML round-trip preserves pipeline structure.\n")


def run_agent(pipeline: Pipeline, query: str) -> None:
    """Run the agent with a query (requires live API keys)."""
    print(f"Query: {query}\n")
    result = pipeline.run(data={"agent": {"messages": [ChatMessage.from_user(query)]}})
    print("--- Agent response ---")
    print(result["agent"]["last_message"].text)


if __name__ == "__main__":
    pipeline = build_pipeline()
    restored = roundtrip_yaml(pipeline)
    verify_roundtrip(pipeline, restored)

    run_agent(
        restored,
        "Write a Python one-liner to /tmp/hello.py that prints 'Hello from E2B!', run it, then show me the output.",
    )
