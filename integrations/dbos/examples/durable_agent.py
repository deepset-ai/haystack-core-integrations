# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
A durable Agent backed by SQLite, and what recovery looks like.

Run it twice:

    python examples/durable_agent.py         # crashes halfway through
    python examples/durable_agent.py --resume  # recovers, without re-issuing the completed model calls

The first run raises after the Agent has finished, leaving the workflow in the ERROR state. The second run recovers
it with `DBOS.fork_workflow`, which re-executes the workflow body while completed steps return their recorded
output - so the model is not called again for the turns that already happened.

Needs `OPENAI_API_KEY`. State is written to `dbos.sqlite` in the working directory; delete it to start over.
"""

import sys

from dbos import DBOS, DBOSConfig, SetWorkflowID
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import tool

from haystack_integrations.dbos import durable_agent

WORKFLOW_ID = "durable-agent-example"


@tool
def city_population(city: str) -> str:
    """Return the population of a city."""
    known = {"berlin": "3.9 million", "paris": "2.1 million", "rome": "2.8 million"}
    print(f"  [tool] city_population({city!r}) - this runs again on recovery, so keep tools idempotent")
    return known.get(city.lower(), "unknown")


agent = durable_agent(
    Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[city_population],
        system_prompt="Answer with a single sentence.",
        tool_concurrency_limit=1,
    ),
    name="population_agent",
)

crash_after_answering = "--resume" not in sys.argv


@DBOS.workflow()
def answer(question: str) -> str:
    result = agent.run(messages=[ChatMessage.from_user(question)])
    if crash_after_answering:
        msg = "simulated crash after the Agent produced its answer"
        raise RuntimeError(msg)
    return result["last_message"].text


def main() -> None:
    config: DBOSConfig = {"name": "haystack-durable-agent", "system_database_url": "sqlite:///dbos.sqlite"}
    DBOS(config=config)
    DBOS.launch()

    question = "Which is bigger, Berlin or Rome?"

    if crash_after_answering:
        print("First run: the Agent answers, then the process 'crashes'.")
        try:
            with SetWorkflowID(WORKFLOW_ID):
                answer(question)
        except RuntimeError as error:
            print(f"  crashed: {error}")
        print("\nNow run again with --resume. The model will not be called for the turns already recorded.")
        return

    print("Recovering the workflow. Watch for the absence of new model calls.")
    recovered = DBOS.fork_workflow(WORKFLOW_ID, 2)
    print(f"  answer: {recovered.get_result()}")

    steps = DBOS.list_workflow_steps(WORKFLOW_ID)
    print(f"  recorded steps in the original run: {[step['function_name'] for step in steps]}")


if __name__ == "__main__":
    main()
