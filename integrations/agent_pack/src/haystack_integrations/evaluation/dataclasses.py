# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any

from haystack.dataclasses import ChatMessage

ToolNames = str | tuple[str, ...] | list[str]


@dataclass
class ToolRunStats:
    """
    The tool calls one agent run made, and what they add up to.

    :param calls: Every call the run made, in the order it made them, as `(tool name, the arguments it passed)`,
        for example `[("list_metadata_fields", {}), ("search_documents", {"query": "CRISPR", "filters": None})]`.
    :param errors: The calls that came back an error, as `(tool name, what it said)`, for example
        `[("get_metadata_field_values", "field 'nope' does not exist in the store")]`.
    """

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ToolRunStats":
        """
        Collect the tool calls and error results of one agent run.

        :param messages: The messages returned by `agent.run(...)`.
        :returns: The statistics for that run.
        """
        return cls(
            calls=[(call.tool_name, call.arguments or {}) for message in messages for call in message.tool_calls],
            errors=[
                (result.origin.tool_name, str(result.result))
                for message in messages
                for result in message.tool_call_results
                if result.error
            ],
        )

    @staticmethod
    def _named(tools: ToolNames) -> set[str]:
        """
        Normalize one tool name or a group of them to a set.

        :param tools: A tool name, or several of them.
        :returns: The names as a set.
        """
        return {tools} if isinstance(tools, str) else set(tools)

    def calls_to(self, tools: ToolNames) -> int:
        """
        Count the calls made to one tool, or to any of a group of them.

        :param tools: A tool name, or several of them.
        :returns: How many calls the run made to them.
        """
        wanted = self._named(tools=tools)
        return sum(1 for name, _ in self.calls if name in wanted)

    def calls_with_argument(self, tools: ToolNames, argument: str) -> int:
        """
        Count the calls that passed a non-empty value for one argument, such as a retrieval carrying a filter.

        :param tools: A tool name, or several of them.
        :param argument: The argument that has to carry a value.
        :returns: How many of those calls passed something for it.
        """
        wanted = self._named(tools=tools)
        return sum(1 for name, arguments in self.calls if name in wanted and arguments.get(argument))

    def called_before(self, tools: ToolNames, other: ToolNames) -> bool:
        """
        Whether the run reached for one tool before it reached for another.

        :param tools: The tool, or tools, that should come first.
        :param other: The tool, or tools, they should come before.
        :returns: True when one of `tools` was called and none of `other` was called before it. False when
            `other` came first, and when neither was called at all.
        """
        wanted, after = self._named(tools=tools), self._named(tools=other)
        for name, _ in self.calls:
            if name in wanted:
                return True
            if name in after:
                return False
        return False


@dataclass(kw_only=True)
class RetrievalEvalCase:
    """
    One labelled question and the documents needed to answer it.

    :param question: The question to put to whatever is under evaluation.
    :param evidence: Ground truth, as `{document id: the quote found in that document}`. The keys are the
        documents recall is measured against, and the values say what each one was needed for, for example
        `{"a1b2c3...": "Tyreek Hill now needs to ...", "d4e5f6...": "The Dolphins went on to ..."}`.
        A harness that knows which documents are needed but not what they were needed for leaves the values empty.
    :param min_recall: Minimum share of the needed documents that must be found.
    :param min_precision: Minimum share of what came back that must be needed. Left at 0 by default, because
        returning more than was asked for is not itself a fault; raise it to make over-retrieval cost something.
    """

    question: str
    evidence: dict[str, str] = field(default_factory=dict)
    min_recall: float = 1.0
    min_precision: float = 0.0

    def __post_init__(self) -> None:
        """Require ground truth to score against."""
        if not self.evidence:
            msg = f"Case {self.question!r} needs evidence to score against."
            raise ValueError(msg)

    @property
    def expected_document_ids(self) -> frozenset[str]:
        """The documents needed to answer a question based on the provided evidence"""
        return frozenset(self.evidence)

    def found_at(self, document_ids: list[str], k: int | None = None) -> frozenset[str]:
        """
        Return which of the needed documents a run returned within its first `k`.

        :param document_ids: What the run returned, in the order it ranked them.
        :param k: Rank cutoff, or `None` to count everything returned.
        :returns: The needed documents that were found.
        """
        # Deduplicated in the returned order first, since which documents fall past the cutoff depends on how
        # the run ranked them rather than on how many times each was returned.
        ranked = list(dict.fromkeys(document_ids))
        return self.expected_document_ids & set(ranked[:k] if k is not None else ranked)

    def recall_at(self, document_ids: list[str], k: int | None = None) -> float:
        """
        Return the share of the needed documents a run found within its first `k`.

        :param document_ids: What the run returned, in the order it ranked them.
        :param k: Rank cutoff, or `None` to count everything returned.
        :returns: Recall@k, or 0.0 when the eval case names no documents.
        """
        if not self.evidence:
            return 0.0
        return len(self.found_at(document_ids=document_ids, k=k)) / len(self.evidence)

    def precision_at(self, document_ids: list[str], k: int | None = None) -> float:
        """
        Return the share of a run's first `k` documents that were needed.

        :param document_ids: What the run returned, in the order it ranked them.
        :param k: Rank cutoff, or `None` to count everything returned.
        :returns: Precision@k, or 0.0 when the run returned nothing.
        """
        ranked = list(dict.fromkeys(document_ids))
        scored = ranked[:k] if k is not None else ranked
        if not scored:
            return 0.0
        return len(self.found_at(document_ids=document_ids, k=k)) / len(scored)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the eval case into a dictionary.

        :returns: A dictionary with one key per field, JSON compatible and with the evidence in a stable order,
            so an evaluation set is identified the same way whichever order it was built in.
        """
        data = asdict(self)
        data["evidence"] = dict(sorted(self.evidence.items()))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalEvalCase":
        """
        Create a new eval case from a dictionary.

        :param data: The dictionary to build the eval case from.
        :returns: The created object.
        """
        return cls(**data)


@dataclass(kw_only=True)
class RAGEvalCase(RetrievalEvalCase):
    """
    One eval case for a RAG agent, which answers using tools and so has budgets for them.

    :param tool_budgets: How many times the run may call a tool, or a group of tools sharing one allowance, as
        `{tool name or names: limit}`. Which tools an agent has is the agent's business, so the caller names
        them; anything left unnamed falls to the `*` entry, or to a default. See `resolve_tool_budgets`.
    :param max_tool_errors: Tolerated failing tool calls.
    :param max_steps: Cap on agent steps, or `None` to leave the agent's own cap as the only one.
    :param require_citations: Whether an answer grounded in retrieved documents must cite at least one of them.
        An answer with no citations at all otherwise passes a citation check trivially.
    """

    tool_budgets: dict[ToolNames, int] = field(default_factory=dict)
    max_tool_errors: int = 0
    max_steps: int | None = None
    require_citations: bool = True

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the eval case into a dictionary.

        :returns: A dictionary with one key per field, with the tool budgets as `[names, limit]` pairs ordered by
            name, since a group of tool names is a tuple and JSON has no such key.
        """
        data = super().to_dict()
        data["tool_budgets"] = sorted(
            ([(tools if isinstance(tools, str) else list(tools)), limit] for tools, limit in self.tool_budgets.items()),
            key=lambda entry: str(entry[0]),
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGEvalCase":
        """
        Create a new eval case from a dictionary.

        :param data: The dictionary to build the eval case from.
        :returns: The created object.
        """
        arguments = dict(data)
        budgets = arguments.get("tool_budgets")
        if isinstance(budgets, list):
            arguments["tool_budgets"] = {
                (tools if isinstance(tools, str) else tuple(tools)): limit for tools, limit in budgets
            }
        return cls(**arguments)


@dataclass(kw_only=True)
class ModelTokenUsage:
    """
    Raw token usage attributable to one model deployment.

    :param input_tokens: Number of input tokens consumed by the model.
    :param output_tokens: Number of output tokens generated by the model.
    """

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(kw_only=True)
class ReportedUsage:
    """
    Token usage one LLM call reported.

    :param model: The model identifier the call reported, or `None` when it reported none, which leaves the
        eval case unpriceable.
    :param tokens: The token counts the call reported, or `None` when it reported neither an input nor an
        output count, which also leaves the eval case unpriceable.
    """

    model: str | None = None
    tokens: ModelTokenUsage | None = None


@dataclass(kw_only=True)
class EvalCaseSummary:
    """
    Summarizes what an eval case's spans reported about its token usage and per-stage outputs.

    :param models: Token usage attributed to each model the eval case called, keyed by model identifier.
    :param outputs: How many items each component emitted, by component name and output socket.
    :param texts: A sample of whatever each component emitted as text, by component name and output socket,
        capped by the tracer that recorded it.
    :param all_tokens_reported: Whether every LLM call reported its token counts. False means the token usage
        below is underestimated, so the eval case must not be priced.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    all_tokens_reported: bool = True


@dataclass(kw_only=True)
class EvalMetrics:
    """
    Metrics produced by a harness evaluator for one target configuration.

    :param quality: Normalized aggregate quality score in the inclusive range `[0.0, 1.0]`. Each harness evaluator
        defines which checks contribute to this score.
    :param latency_ms: Mean end-to-end evaluation latency in milliseconds.
    :param model_usage: Raw token usage keyed by model identifier.
    :param cost: Cost derived from `model_usage`, or `None` when usage has not been priced or includes an unknown model.
    :param eval_cases: One record per eval case measured, each in whatever shape its evaluator scores. Empty for
        an evaluator that reports only an aggregate.
    :param details: Evaluator-specific metrics and diagnostic information.
    """

    quality: float
    latency_ms: float
    model_usage: dict[str, ModelTokenUsage] = field(default_factory=dict)
    cost: float | None = None
    eval_cases: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the shared normalized quality contract."""
        if not 0.0 <= self.quality <= 1.0:
            msg = "quality must be between 0.0 and 1.0."
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "model_usage": {model: asdict(obj=usage) for model, usage in self.model_usage.items()},
            "cost": self.cost,
            "eval_cases": self.eval_cases,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalMetrics":
        """
        Restore metrics from a serialized representation.

        :param data: Serialized eval metrics.
        :returns: The restored eval metrics.
        """
        cost = data.get("cost")
        return cls(
            quality=float(data["quality"]),
            latency_ms=float(data["latency_ms"]),
            model_usage={model: ModelTokenUsage(**usage) for model, usage in (data.get("model_usage") or {}).items()},
            cost=None if cost is None else float(cost),
            eval_cases=data.get("eval_cases") or [],
            details=data.get("details") or {},
        )
