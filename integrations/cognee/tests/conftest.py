# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# Set before cognee is imported so setup_logging() picks up WARNING level.
os.environ.setdefault("LOG_LEVEL", "WARNING")

_NOISY_LOGGERS = ("aiosqlite", "sqlalchemy", "sqlalchemy.engine", "alembic")


def _silence_noisy_loggers() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_noisy_loggers()
