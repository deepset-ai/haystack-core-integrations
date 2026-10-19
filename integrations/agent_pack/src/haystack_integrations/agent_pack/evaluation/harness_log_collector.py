# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

# The maximum length of a message to keep.
MAX_MESSAGE_CHARS = 400

# The maximum number of distinct messages to keep.
MAX_DISTINCT_MESSAGES = 12


@dataclass
class CollectedLogs:
    """
    Distinct diagnostics emitted during one evaluation, with how often each occurred.

    :param counts: How many times each distinct diagnostic occurred, keyed by its level, the logger that
        emitted it, and its rendered message cut to `MAX_MESSAGE_CHARS`.
    :param dropped: How many diagnostics arrived after `MAX_DISTINCT_MESSAGES` distinct ones were already held,
        so a reader knows the listing is partial.
    :param lock: Guards the counts and dropped, since what is being measured may log from several threads or tasks.
    """

    counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    dropped: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, level: str, logger_name: str, message: str) -> None:
        """
        Record one diagnostic, collapsing repeats of the same message.

        :param level: The record's level name.
        :param logger_name: The logger that emitted it.
        :param message: The rendered message.
        """
        key = (level, logger_name, message[:MAX_MESSAGE_CHARS])
        with self.lock:
            if key not in self.counts and len(self.counts) >= MAX_DISTINCT_MESSAGES:
                self.dropped += 1
                return
            self.counts[key] += 1

    def to_list(self) -> list[dict[str, Any]]:
        """
        Return the diagnostics as JSON-compatible records, most frequent first.

        :returns: One entry per distinct message, carrying its level, logger, text and count.
        """
        # We use a lock here to ensure that the counts are not modified while we are iterating over them.
        with self.lock:
            return [
                {"level": level, "logger": logger_name, "message": message, "count": count}
                for (level, logger_name, message), count in self.counts.most_common()
            ]


class _CollectingHandler(logging.Handler):
    def __init__(self, logs: CollectedLogs) -> None:
        super().__init__(level=logging.WARNING)
        self.logs = logs

    def emit(self, record: logging.LogRecord) -> None:
        """Record one rendered message, never letting a logging failure disturb the run being measured."""
        try:
            self.logs.add(level=record.levelname, logger_name=record.name, message=record.getMessage())
        except Exception:
            self.handleError(record)


class HarnessLogCollector:
    """Capture the warnings and errors whatever a harness runs emits, for the duration of one evaluation."""

    def __init__(self, logger_names: tuple[str, ...] = ("haystack", "haystack_integrations")) -> None:
        """
        Create a collector.

        :param logger_names: Roots of the logger trees to listen to. By default, they are "haystack" and
            "haystack_integrations", which cover anything Haystack runs: e.g. pipelines, components, tools, hooks, etc.
        """
        self.logger_names = logger_names

    @contextmanager
    def collect(self) -> Iterator[CollectedLogs]:
        """
        Listen to the configured logger trees, restoring each one's level afterward.

        :returns: The diagnostics collected while the block ran.
        """
        logs = CollectedLogs()
        handler = _CollectingHandler(logs=logs)
        touched: list[tuple[logging.Logger, int]] = []
        try:
            for name in self.logger_names:
                logger = logging.getLogger(name)
                touched.append((logger, logger.level))
                # A tree left at its default would filter warnings before any handler sees them; anything already
                # more verbose is left alone so a caller's own logging setup survives the evaluation.
                if not logger.isEnabledFor(logging.WARNING):
                    logger.setLevel(logging.WARNING)
                logger.addHandler(handler)
            yield logs
        finally:
            for logger, level in touched:
                logger.removeHandler(handler)
                logger.setLevel(level)
