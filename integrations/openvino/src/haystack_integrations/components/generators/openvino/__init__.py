# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .chat.chat_generator import OpenVINOChatGenerator
from .generator import OpenVINOGenerator

__all__ = ["OpenVINOGenerator", "OpenVINOChatGenerator"]
