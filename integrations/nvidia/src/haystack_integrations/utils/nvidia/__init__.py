# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .nim_backend import NimBackend
from .statics import Model
from .utils import is_hosted, url_validation

__all__ = ["Model", "NimBackend", "is_hosted", "url_validation"]
