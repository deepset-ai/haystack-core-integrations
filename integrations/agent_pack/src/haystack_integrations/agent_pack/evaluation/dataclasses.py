# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from typing import Any

from .tool_run_stats import ToolNames


@dataclass(kw_only=True)
class EvalCase:
    """
    What every harness needs to score one labelled question: the question, and the documents that answer it.

    Harnesses subclass this and add what only they can measure.

    :param question: The question to put to whatever is under evaluation.
    :param evidence: Ground truth, as `{document id: the quote found in that document}`. The keys are the
        documents recall is measured against, and the values say what each one was needed for:

            {"a1b2c3...": "Tyreek Hill now needs to ...", "d4e5f6...": "The Dolphins went on to ..."}

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
        """The documents an answer needs, which are the ones its evidence was found in."""
        return frozenset(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the eval case into a dictionary.

        :returns: A dictionary with one key per field, JSON compatible and stable enough to identify an
            evaluation set.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        """
        Create a new eval case from a dictionary.

        :param data: The dictionary to build the eval case from.
        :returns: The created object.
        """
        return cls(**data)


@dataclass(kw_only=True)
class RAGEvalCase(EvalCase):
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
