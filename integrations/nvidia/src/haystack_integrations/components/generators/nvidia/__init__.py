# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .chat.chat_generator import NvidiaChatGenerator
from .generator import NvidiaGenerator

__all__ = ["NvidiaChatGenerator", "NvidiaGenerator"]
