# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.hooks.human_in_the_loop import ConfirmationUIResult, ToolExecutionDecision
from haystack.hooks.human_in_the_loop.strategies import (
    MODIFICATION_FEEDBACK_TEMPLATE,
    REJECTION_FEEDBACK_TEMPLATE,
    USER_FEEDBACK_TEMPLATE,
)
from haystack.hooks.human_in_the_loop.types import ConfirmationPolicy
from haystack.utils.deserialization import deserialize_component_inplace

from dbos import DBOS
from haystack_integrations.dbos._runtime import in_workflow

logger = logging.getLogger(__name__)

DEFAULT_EVENT_KEY = "haystack.pending_tool_call"
DEFAULT_TIMEOUT_SECONDS = 86400.0


def _to_ui_result(message: Any) -> ConfirmationUIResult:
    """
    Turn a received DBOS message into a `ConfirmationUIResult`.

    Accepts either a plain action string such as `"confirm"` or a dictionary with `action` and the optional
    `feedback` and `new_tool_params` keys.
    """
    if isinstance(message, ConfirmationUIResult):
        return message
    if isinstance(message, str):
        return ConfirmationUIResult(action=message)
    if isinstance(message, dict) and isinstance(message.get("action"), str):
        return ConfirmationUIResult(
            action=message["action"],
            feedback=message.get("feedback"),
            new_tool_params=message.get("new_tool_params"),
        )
    msg = (
        "A tool confirmation message must be an action string, a ConfirmationUIResult, or a dictionary with an "
        f"'action' key. Received {message!r}."
    )
    raise ValueError(msg)


def _to_decision(
    ui_result: ConfirmationUIResult,
    *,
    tool_name: str,
    tool_call_id: str | None,
    tool_params: dict[str, Any],
    reject_template: str,
    modify_template: str,
    user_feedback_template: str,
) -> ToolExecutionDecision:
    """Apply the same confirm/reject/modify semantics that `BlockingConfirmationStrategy` uses."""
    if ui_result.action == "reject":
        explanation = reject_template.format(tool_name=tool_name)
        if ui_result.feedback:
            explanation += " " + user_feedback_template.format(feedback=ui_result.feedback)
        return ToolExecutionDecision(
            tool_name=tool_name, execute=False, tool_call_id=tool_call_id, feedback=explanation
        )
    if ui_result.action == "modify" and ui_result.new_tool_params:
        final_params = dict(ui_result.new_tool_params)
        explanation = modify_template.format(tool_name=tool_name, final_tool_params=final_params)
        if ui_result.feedback:
            explanation += " " + user_feedback_template.format(feedback=ui_result.feedback)
        return ToolExecutionDecision(
            tool_name=tool_name,
            execute=True,
            tool_call_id=tool_call_id,
            feedback=explanation,
            final_tool_params=final_params,
        )
    return ToolExecutionDecision(
        tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
    )


class DBOSConfirmationStrategy:
    """
    A confirmation strategy that suspends a DBOS workflow until an approval arrives.

    Register it on Haystack's `ConfirmationHook`. When a tool needs confirmation, the strategy publishes the pending
    call as a DBOS workflow event and then waits on `DBOS.recv`. The workflow can stay suspended for as long as the
    timeout allows and survives a process restart while it waits, so the approval can come from anywhere - an HTTP
    handler, a chat bot, a command line tool.

    Because `set_event` and `recv` are themselves checkpointed, a workflow that is recovered after a crash re-reads
    the decision it already received instead of asking again.

    ```python
    from haystack.hooks.human_in_the_loop import ConfirmationHook
    from haystack_integrations.dbos import DBOSConfirmationStrategy

    hook = ConfirmationHook(confirmation_strategies={"delete_file": DBOSConfirmationStrategy()})
    ```

    The approver reads the pending call and answers on the tool's topic:

    ```python
    pending = DBOS.get_event(workflow_id, "haystack.pending_tool_call", timeout_seconds=0)
    DBOS.send(workflow_id, {"action": "confirm"}, topic="confirm.delete_file")
    ```

    A message may be an action string (`"confirm"`, `"reject"`, `"modify"`) or a dictionary with `action` and the
    optional `feedback` and `new_tool_params` keys.
    """

    def __init__(
        self,
        *,
        topic: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        event_key: str = DEFAULT_EVENT_KEY,
        confirmation_policy: ConfirmationPolicy | None = None,
        reject_template: str = REJECTION_FEEDBACK_TEMPLATE,
        modify_template: str = MODIFICATION_FEEDBACK_TEMPLATE,
        user_feedback_template: str = USER_FEEDBACK_TEMPLATE,
    ) -> None:
        """
        Initialize the strategy.

        :param topic:
            The DBOS topic to wait on. Defaults to `confirm.{tool_name}`, which gives each tool its own topic so
            that confirmations requested in the same Agent step cannot consume one another's messages.
        :param timeout_seconds:
            How long to wait for a decision before giving up. Defaults to 24 hours. On timeout the tool is not
            executed and the model is told the confirmation expired.
        :param event_key:
            The DBOS event key under which the pending tool call is published.
        :param confirmation_policy:
            Optional policy deciding which calls need confirmation. Without one, every call is confirmed.
        :param reject_template:
            Template for rejection feedback. It should include a `{tool_name}` placeholder.
        :param modify_template:
            Template for modification feedback. It should include `{tool_name}` and `{final_tool_params}`.
        :param user_feedback_template:
            Template for the approver's own feedback. It should include a `{feedback}` placeholder.
        """
        self.topic = topic
        self.timeout_seconds = timeout_seconds
        self.event_key = event_key
        self.confirmation_policy = confirmation_policy
        self.reject_template = reject_template
        self.modify_template = modify_template
        self.user_feedback_template = user_feedback_template

    def _topic_for(self, tool_name: str) -> str:
        return self.topic or f"confirm.{tool_name}"

    def _should_ask(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> bool:
        if self.confirmation_policy is None:
            return True
        return self.confirmation_policy.should_ask(
            tool_name=tool_name, tool_description=tool_description, tool_params=tool_params
        )

    def _approved(self, tool_name: str, tool_call_id: str | None, tool_params: dict[str, Any]) -> ToolExecutionDecision:
        return ToolExecutionDecision(
            tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
        )

    def _pending_event(self, tool_name: str, tool_description: str, tool_params: dict[str, Any]) -> dict[str, Any]:
        return {"tool_name": tool_name, "tool_description": tool_description, "tool_params": tool_params}

    def _require_workflow(self, tool_name: str) -> bool:
        """Return whether a decision can be awaited, warning once when it cannot."""
        if in_workflow():
            return True
        logger.warning(
            "DBOSConfirmationStrategy needs a DBOS workflow to wait for a decision, but '{tool_name}' was confirmed "
            "outside one. The tool call is allowed to proceed. Call the Agent from inside a @DBOS.workflow().",
            tool_name=tool_name,
        )
        return False

    def _timed_out(self, tool_name: str, tool_call_id: str | None) -> ToolExecutionDecision:
        logger.warning(
            "No confirmation for '{tool_name}' arrived within {timeout} seconds; the tool call was not executed.",
            tool_name=tool_name,
            timeout=self.timeout_seconds,
        )
        return ToolExecutionDecision(
            tool_name=tool_name,
            execute=False,
            tool_call_id=tool_call_id,
            feedback=f"Confirmation for tool '{tool_name}' was not given within {self.timeout_seconds} seconds.",
        )

    def _decide(
        self,
        message: Any,
        *,
        tool_name: str,
        tool_description: str,
        tool_call_id: str | None,
        tool_params: dict[str, Any],
    ) -> ToolExecutionDecision:
        if message is None:
            return self._timed_out(tool_name, tool_call_id)
        ui_result = _to_ui_result(message)
        if self.confirmation_policy is not None:
            self.confirmation_policy.update_after_confirmation(tool_name, tool_description, tool_params, ui_result)
        return _to_decision(
            ui_result,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_params=tool_params,
            reject_template=self.reject_template,
            modify_template=self.modify_template,
            user_feedback_template=self.user_feedback_template,
        )

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ToolExecutionDecision:
        """
        Wait for a decision on a pending tool call.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters the model wants to call the tool with.
        :param tool_call_id: Optional identifier correlating the decision with a specific tool invocation.
        :param confirmation_strategy_context: Unused; decisions arrive over DBOS rather than a per-request resource.
        :returns: The decision on whether to execute the tool, and with which parameters.
        """
        if not self._should_ask(tool_name, tool_description, tool_params) or not self._require_workflow(tool_name):
            return self._approved(tool_name, tool_call_id, tool_params)
        DBOS.set_event(self.event_key, self._pending_event(tool_name, tool_description, tool_params))
        message = DBOS.recv(self._topic_for(tool_name), timeout_seconds=self.timeout_seconds)
        return self._decide(
            message,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_call_id=tool_call_id,
            tool_params=tool_params,
        )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> ToolExecutionDecision:
        """
        Async version of `run`, which waits without blocking the event loop.

        :param tool_name: The name of the tool to be executed.
        :param tool_description: The description of the tool.
        :param tool_params: The parameters the model wants to call the tool with.
        :param tool_call_id: Optional identifier correlating the decision with a specific tool invocation.
        :param confirmation_strategy_context: Unused; decisions arrive over DBOS rather than a per-request resource.
        :returns: The decision on whether to execute the tool, and with which parameters.
        """
        if not self._should_ask(tool_name, tool_description, tool_params) or not self._require_workflow(tool_name):
            return self._approved(tool_name, tool_call_id, tool_params)
        await DBOS.set_event_async(self.event_key, self._pending_event(tool_name, tool_description, tool_params))
        message = await DBOS.recv_async(self._topic_for(tool_name), timeout_seconds=self.timeout_seconds)
        return self._decide(
            message,
            tool_name=tool_name,
            tool_description=tool_description,
            tool_call_id=tool_call_id,
            tool_params=tool_params,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the strategy to a dictionary.

        :returns: Dictionary with serialized data.
        """
        policy = self.confirmation_policy
        return default_to_dict(
            self,
            topic=self.topic,
            timeout_seconds=self.timeout_seconds,
            event_key=self.event_key,
            confirmation_policy=policy.to_dict() if policy is not None else None,
            reject_template=self.reject_template,
            modify_template=self.modify_template,
            user_feedback_template=self.user_feedback_template,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DBOSConfirmationStrategy":
        """
        Deserialize the strategy from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized strategy.
        """
        if data.get("init_parameters", {}).get("confirmation_policy") is not None:
            deserialize_component_inplace(data["init_parameters"], key="confirmation_policy")
        return default_from_dict(cls, data)
