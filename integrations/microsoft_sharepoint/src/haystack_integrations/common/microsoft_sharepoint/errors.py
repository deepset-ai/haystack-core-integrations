# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class SharePointError(Exception):
    """Base class for errors raised by the Microsoft SharePoint integration."""


class SharePointConfigError(SharePointError):
    """Raised when a Microsoft SharePoint component is misconfigured."""


class SharePointRequestError(SharePointError):
    """Raised when a Microsoft Graph request fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
