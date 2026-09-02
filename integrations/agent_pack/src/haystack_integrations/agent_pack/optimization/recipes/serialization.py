# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Identifying a recipe by its content."""

import hashlib
import json

from haystack_integrations.agent_pack.optimization.recipes.types.protocol import CandidateRecipe


def recipe_fingerprint(recipe: CandidateRecipe) -> str:
    """
    Return a stable content hash for a typed candidate recipe.

    :param recipe: The recipe to hash.
    :returns: A hex SHA-256 digest of the recipe's canonical serialized form.
    """
    payload = json.dumps(recipe.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
