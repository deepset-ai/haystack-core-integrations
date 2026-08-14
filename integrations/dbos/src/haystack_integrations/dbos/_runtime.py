# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for deciding whether the current call can be checkpointed by DBOS."""

import asyncio
from typing import Any

from dbos import DBOS

# `DBOS.workflow_id` and `DBOS.step_status` are safe to read before `DBOS()` is constructed: both return None
# rather than raising, so these checks work in applications that never launch DBOS at all.


def in_workflow() -> bool:
    """Return whether the caller is running inside a DBOS workflow."""
    return DBOS.workflow_id is not None


def in_step() -> bool:
    """Return whether the caller is already running inside a DBOS step."""
    return DBOS.step_status is not None


def checkpointing() -> bool:
    """
    Return whether a call made right now would be checkpointed.

    False outside a workflow (no DBOS, or an ordinary call path) and False inside a step, where DBOS collapses a
    nested step into its caller instead of recording it separately.
    """
    return in_workflow() and not in_step()


async def run_component_async(component: Any, **inputs: Any) -> dict[str, Any]:
    """
    Await a component's `run_async` if it has one, otherwise run its `run` in a worker thread.

    This mirrors what the Agent does for components that are sync-only.
    """
    run_async = getattr(component, "run_async", None)
    if run_async is not None:
        result = await run_async(**inputs)
        return dict(result)
    return dict(await asyncio.to_thread(component.run, **inputs))
