# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .nim_backend import Model, NimBackend
from .utils import is_hosted, url_validation

__all__ = ["NimBackend", "Model", "is_hosted", "url_validation"]
