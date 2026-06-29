# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .text_router import TransformersTextRouter
from .zero_shot_text_router import TransformersZeroShotTextRouter

__all__ = ["TransformersTextRouter", "TransformersZeroShotTextRouter"]
