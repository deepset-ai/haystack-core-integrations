# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any

from haystack import logging
from haystack.components.agents import Agent
from haystack.tools import SearchableToolset, Toolset

from dbos import StepOptions
from haystack_integrations.dbos.chat_generator import DBOSChatGenerator

logger = logging.getLogger(__name__)


# Init parameters the Agent does not store under their own name, most-preferred attribute first. In haystack-ai
# 3.0.0 `Agent.state_schema` holds the resolved schema, which includes reserved keys that cannot be passed back to
# `__init__`; the value that was passed in lives in `_state_schema`.
_INIT_ATTRIBUTES: dict[str, tuple[str, ...]] = {"state_schema": ("_state_schema", "state_schema")}


def _init_value(agent: Agent, name: str) -> Any:
    """Read back the value an Agent was constructed with for the init parameter `name`."""
    for attribute in _INIT_ATTRIBUTES.get(name, (name,)):
        if hasattr(agent, attribute):
            return getattr(agent, attribute)
    msg = f"Cannot rebuild {type(agent).__name__}: it does not expose its '{name}' init parameter."
    raise AttributeError(msg)


def _rebuild(agent: Agent, **overrides: Any) -> Agent:
    """
    Return a copy of `agent` with the given init parameters replaced.

    Uses `Agent.clone` where available and falls back to rebuilding from the init signature, which is what `clone`
    itself does. The fallback exists because `clone` is not in every supported `haystack-ai` release.
    """
    clone = getattr(agent, "clone", None)
    if callable(clone):
        return clone(**overrides)  # type: ignore[no-any-return]
    init_params = inspect.signature(type(agent).__init__).parameters
    params: dict[str, Any] = {name: _init_value(agent, name) for name in init_params if name != "self"}
    return type(agent)(**{**params, **overrides})


def _warn_on_dynamic_toolsets(agent: Agent) -> None:
    """
    Warn when the Agent carries a toolset whose contents are discovered while the Agent runs.

    The Agent re-reads its tools on every step, so a toolset that fills itself in as a side effect of running one of
    its own tools will be empty on recovery: the discovery call is replayed from its checkpoint without running its
    body. The replayed model message then asks for a tool that is no longer registered.
    """
    tools = agent.tools
    candidates = tools if isinstance(tools, list) else [tools]
    for candidate in candidates:
        if isinstance(candidate, SearchableToolset):
            logger.warning(
                "Agent uses a SearchableToolset, whose tools are discovered while the Agent runs. Tool discovery is "
                "not replayed on recovery, so a recovered run can fail with a missing tool. Prefer a static tool "
                "list for durable Agents."
            )
        elif isinstance(candidate, Toolset) and not list(candidate.tools):
            logger.warning(
                "Agent uses a Toolset of type '{toolset}' that is empty at construction time. If it fills itself in "
                "while the Agent runs, its tools will be missing on recovery. Prefer a static tool list for durable "
                "Agents.",
                toolset=type(candidate).__name__,
            )


def durable_agent(
    agent: Agent,
    *,
    name: str | None = None,
    step_options: StepOptions | None = None,
) -> Agent:
    """
    Return a copy of `agent` whose model calls are checkpointed as DBOS steps.

    The returned Agent behaves exactly like the one passed in, except that each call to its Chat Generator is
    recorded when the Agent runs inside a `@DBOS.workflow()`. Write the workflow yourself and call the Agent from
    inside it:

    ```python
    from dbos import DBOS, DBOSConfig
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.dbos import durable_agent

    agent = durable_agent(Agent(chat_generator=OpenAIChatGenerator()), name="support_agent")


    @DBOS.workflow()
    def answer(question: str) -> str:
        return agent.run(messages=[ChatMessage.from_user(question)])["last_message"].text


    config: DBOSConfig = {"name": "support", "system_database_url": "sqlite:///dbos.sqlite"}
    DBOS(config=config)
    DBOS.launch()
    ```

    Tool calls are not checkpointed. On recovery the workflow body runs again and every tool call made before the
    crash is executed a second time, so tools must either be idempotent or be steps themselves. To make a tool
    durable, decorate its function with `@DBOS.step()` and run the Agent with `run_async`, or with `run` and
    `tool_concurrency_limit=1`: synchronous tool execution uses a thread pool that shares the DBOS step counter
    across workers, which makes concurrent tool steps unsafe.

    :param agent:
        The Agent to make durable. It is not modified.
    :param name:
        Prefix for the recorded step name. Give each Agent in an application its own name. Defaults to the Chat
        Generator's class name in snake case.
    :param step_options:
        Options forwarded to DBOS for the model-call step, such as `retries_allowed` and `max_attempts`.
    :returns:
        A new Agent with its Chat Generator wrapped in a `DBOSChatGenerator`.
    """
    if isinstance(agent.chat_generator, DBOSChatGenerator):
        msg = "The Agent's chat generator is already a DBOSChatGenerator; durable_agent must not be applied twice."
        raise ValueError(msg)

    _warn_on_dynamic_toolsets(agent)

    wrapped = DBOSChatGenerator(agent.chat_generator, name=name, step_options=step_options)
    return _rebuild(agent, chat_generator=wrapped)
