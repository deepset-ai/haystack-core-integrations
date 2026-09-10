# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric

from haystack_integrations.evaluation.agent_pack.dataclasses import ModelTokenUsage

# The span a harness opens around one eval case. Everything traced under it belongs to that eval case.
EVAL_CASE_SPAN = "haystack.harness.eval_case"

# How many of a socket's strings to keep, and how much of each.
MAX_RECORDED_TEXTS = 8
MAX_RECORDED_TEXT_CHARS = 120


def is_generator_span(operation_name: str, tags: dict[str, Any]) -> bool:
    """
    Decide whether a span is a model call, whose token usage is what it reports.

    :param operation_name: The span's operation name.
    :param tags: The span's tags.
    :returns: True for an agent step, a component-held generator, or a generator run as a pipeline component.
    """
    return operation_name in ("haystack.agent.step.llm", "haystack.chat_generator.run") or (
        operation_name == "haystack.component.run"
        and str(tags.get("haystack.component.type", "")).endswith("ChatGenerator")
    )


def get_component_name(tags: dict[str, Any]) -> str | None:
    """
    Name the component a span belongs to.

    :param tags: The span's tags.
    :returns: The component's name, or `None` for a span that is not a component run.
    """
    return str(tags.get("haystack.component.name") or "") or None


def capped(text: str) -> str:
    """
    Cut one recorded string to its allowance, marking the cut so a reader knows there was more.

    :param text: The string a component emitted.
    :returns: The string, ending in an ellipsis when anything was dropped.
    """
    return text if len(text) <= MAX_RECORDED_TEXT_CHARS else f"{text[:MAX_RECORDED_TEXT_CHARS]}..."


def measure_output(value: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Measure how much a component emitted on each socket, and sample whatever it emitted as short strings.

    :param value: The component's output, by socket.
    :returns: How many items each socket carried, and a capped sample of the sockets carrying only strings.
    """
    sizes: dict[str, int] = {}
    texts: dict[str, list[str]] = {}
    for socket, emitted in value.items():
        # A socket carrying one string or one object emits one item, not none: a router or a prompt builder
        # belongs in the chain as much as a retriever does. A string is characters, not items, so it counts once.
        items = list(emitted) if isinstance(emitted, (list, tuple)) else [emitted]
        sizes[socket] = len(items)
        # Documents and messages are counted and dropped, so nothing long is retained by accident.
        if items and all(isinstance(item, str) for item in items):
            texts[socket] = [capped(text=item) for item in items[:MAX_RECORDED_TEXTS]]
    return sizes, texts


@dataclass
class ReportedUsage:
    """
    Token usage reported by a generator.

    :param model: The model identifier the call reported, or `None` when it reported none.
    :param tokens: The token counts the call reported, under whatever keys it used.
    """

    model: Any = None
    tokens: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanRecord:
    """
    A span record which contains reduced information about a span that ran under an eval case.

    :param span_id: Identifies this span among the records of one eval case.
    :param parent_span_id: The span this one ran under, or `None` for the root of a collection.
    :param component_name: The component the span belongs to, or `None` for a span that is not a component run.
    :param is_generator_span: Whether the span is a model call.
    :param output_sizes: How many items the component emitted, by output socket.
    :param output_texts: A capped sample of the sockets that emitted nothing but short strings.
    :param reported_usage: What each of a generator's replies said it spent.
    :param reported_output: Whether an output tag arrived at all. A generator span without one made a model call
        that nobody can account for.
    """

    span_id: str = field(default_factory=lambda: str(uuid4()))
    parent_span_id: str | None = None
    component_name: str | None = None
    is_generator_span: bool = False
    output_sizes: dict[str, int] = field(default_factory=dict)
    output_texts: dict[str, list[str]] = field(default_factory=dict)
    reported_usage: list[ReportedUsage] = field(default_factory=list)
    reported_output: bool = False


@dataclass
class EvalCaseUsage:
    """
    Summarizes what an eval case's spans reported about its token usage and per-stage outputs.

    :param models: Token usage attributed to each model the eval case called, keyed by model identifier.
    :param outputs: How many items each component emitted, by component name and output socket.
    :param texts: A sample of whatever each component emitted as text, by component name and output socket,
        capped at `MAX_RECORDED_TEXTS` entries of `MAX_RECORDED_TEXT_CHARS`.
    :param complete: Whether every model call reported token usage. False means the total token usage is
        underestimated, so the eval case must not be priced.
    :param calls: How many model calls the eval case made.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    complete: bool = True
    calls: int = 0


def eval_case_usage_from_records(records: list[SpanRecord]) -> EvalCaseUsage:
    """
    Fold what an eval case's spans reported into one measurement.

    :param records: The span records collected under one eval case, in the order their spans ended.
    :returns: The eval case's token usage and per-stage output sizes.
    """
    usage = EvalCaseUsage()
    for record in records:
        # Stage sizes come from every component; a generator's reply count says nothing about how much
        # reached the next stage, and nested generators would collide with their owner's entry.
        if record.component_name is not None and not record.is_generator_span and record.output_sizes:
            usage.outputs[record.component_name] = record.output_sizes
            if record.output_texts:
                usage.texts[record.component_name] = record.output_texts

        if not record.is_generator_span:
            continue
        # A model call that reported nothing, or replied with nothing, spent tokens nobody can account for.
        if not record.reported_output:
            usage.complete = False
            continue
        usage.calls += 1
        if not record.reported_usage:
            usage.complete = False
        for model, tokens in ((entry.model, entry.tokens) for entry in record.reported_usage):
            if not isinstance(model, str) or not all(
                any(isinstance(tokens.get(key), (int, float)) for key in keys)
                for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)
            ):
                usage.complete = False
                continue
            current = usage.models.get(model, ModelTokenUsage())
            usage.models[model] = ModelTokenUsage(
                input_tokens=current.input_tokens + _first_numeric(usage=tokens, keys=_INPUT_TOKEN_KEYS),
                output_tokens=current.output_tokens + _first_numeric(usage=tokens, keys=_OUTPUT_TOKEN_KEYS),
            )
    return usage
