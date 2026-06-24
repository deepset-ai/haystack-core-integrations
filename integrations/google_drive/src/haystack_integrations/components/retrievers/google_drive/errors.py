# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class GoogleDriveError(Exception):
    """Base class for errors raised by the Google Drive integration."""


class GoogleDriveConfigError(GoogleDriveError):
    """Raised when the Google Drive retriever is misconfigured."""


class GoogleDriveRequestError(GoogleDriveError):
    """Raised when a Google Drive API request fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
