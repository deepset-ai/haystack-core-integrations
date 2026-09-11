# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric

from haystack_integrations.evaluation.dataclasses import ModelTokenUsage

# The span a harness opens around one eval case. Everything traced under it belongs to that eval case.
EVAL_CASE_SPAN = "haystack.harness.eval_case"


@dataclass
class ReportedUsage:
    """
    Token usage reported by a generator.

    :param model: The model identifier the call reported, or `None` when it reported none, which leaves the
        eval case unpriceable.
    :param tokens: The token counts the call reported, under whatever keys it used.
    """

    model: str | None = None
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
class SpanRecords:
    """
    Every span record collected under one eval case span.

    :param records: The records, in the order their spans ended.
    :param lock: Guards appends, since a run's components may be traced from several threads or tasks.
    """

    records: list[SpanRecord] = field(default_factory=list)
    lock: LockType = field(default_factory=Lock, repr=False)

    def add(self, record: SpanRecord) -> None:
        """
        Collect one closed span's record.

        :param record: What the span reported.
        """
        with self.lock:
            self.records.append(record)


@dataclass
class EvalCaseSummary:
    """
    Summarizes what an eval case's spans reported about its token usage and per-stage outputs.

    :param models: Token usage attributed to each model the eval case called, keyed by model identifier.
    :param outputs: How many items each component emitted, by component name and output socket.
    :param texts: A sample of whatever each component emitted as text, by component name and output socket,
        capped by the tracer that recorded it.
    :param all_tokens_reported: Whether every LLM call reported its token counts. False means the token usage
        below is underestimated, so the eval case must not be priced.
    :param llm_calls: How many LLM calls the eval case made.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    all_tokens_reported: bool = True
    llm_calls: int = 0


def _eval_case_summary_from_records(records: list[SpanRecord]) -> EvalCaseSummary:
    """
    Summarize what an eval case's spans reported.

    :param records: The span records collected under one eval case, in the order their spans ended.
    :returns: The eval case's token usage and per-stage output sizes.
    """
    models: dict[str, ModelTokenUsage] = {}
    outputs: dict[str, dict[str, int]] = {}
    texts: dict[str, dict[str, list[str]]] = {}
    all_tokens_reported = True
    llm_calls = 0
    for record in records:
        # Stage sizes come from every component; a generator's reply count says nothing about how much
        # reached the next stage, and nested generators would collide with their owner's entry.
        if record.component_name is not None and not record.is_generator_span and record.output_sizes:
            outputs[record.component_name] = record.output_sizes
            if record.output_texts:
                texts[record.component_name] = record.output_texts

        # If not a generator span skip
        if not record.is_generator_span:
            continue

        # This means a generator did not report its output so usage cannot be tracked
        if not record.reported_output:
            all_tokens_reported = False
            continue

        # Counted after the output check, so a call is counted once there is evidence it produced something.
        llm_calls += 1

        # If no usage was reported from the provider we indicate not all token usage is accounted for
        if not record.reported_usage:
            all_tokens_reported = False

        for entry in record.reported_usage:
            # Providers key token counts differently, so any known input key and any known output key will do.
            counted_input_and_output = all(
                any(isinstance(entry.tokens.get(key), (int, float)) for key in keys)
                for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)
            )
            # Usage nobody can attribute to a model, or missing either count, cannot be priced.
            if entry.model is None or not counted_input_and_output:
                all_tokens_reported = False
                continue
            current = models.get(entry.model, ModelTokenUsage())
            models[entry.model] = ModelTokenUsage(
                input_tokens=current.input_tokens + _first_numeric(usage=entry.tokens, keys=_INPUT_TOKEN_KEYS),
                output_tokens=current.output_tokens + _first_numeric(usage=entry.tokens, keys=_OUTPUT_TOKEN_KEYS),
            )
    return EvalCaseSummary(
        models=models, outputs=outputs, texts=texts, all_tokens_reported=all_tokens_reported, llm_calls=llm_calls
    )
