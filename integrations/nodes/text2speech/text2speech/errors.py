# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from haystack.errors import NodeError


class Text2SpeechNodeError(NodeError):
    """Exception for issues that occur in a node of the text2speech module"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)
