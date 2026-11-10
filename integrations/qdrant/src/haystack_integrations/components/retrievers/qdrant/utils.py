# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from qdrant_client.http import models


def _resolve_filters(
    *,
    filter_policy: FilterPolicy,
    init_filters: dict[str, Any] | models.Filter | None,
    runtime_filters: dict[str, Any] | models.Filter | None,
) -> dict[str, Any] | models.Filter | None:
    """Apply the filter policy, allowing native Qdrant filters only with REPLACE."""
    if isinstance(init_filters, models.Filter) or isinstance(runtime_filters, models.Filter):
        if filter_policy == FilterPolicy.MERGE:
            msg = (
                "Native Qdrant filters cannot be used with filter_policy set to MERGE. "
                "Set filter_policy to REPLACE or use Haystack filters instead."
            )
            raise ValueError(msg)
        return runtime_filters or init_filters
    return apply_filter_policy(filter_policy=filter_policy, init_filters=init_filters, runtime_filters=runtime_filters)
