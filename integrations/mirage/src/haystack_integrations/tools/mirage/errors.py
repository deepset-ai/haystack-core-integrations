# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class MirageError(Exception):
    """Base class for all Mirage integration errors."""


class MirageConfigError(MirageError):
    """Raised when a Mirage workspace or mount is misconfigured."""


class MirageCommandNotAllowedError(MirageError):
    """Raised when a command is blocked by the tool's security guard (allowlist / denied paths)."""
