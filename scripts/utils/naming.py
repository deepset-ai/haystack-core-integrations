# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Pure string helpers for integration names and paths."""

import re
from pathlib import Path


def validate_name(name: str, integrations_dir: Path) -> str | None:
    """Return an error message if the name is invalid, None otherwise."""
    if not name:
        return "Name cannot be empty."
    if name != name.lower():
        return "Name must be lowercase."
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        return "Name must start with a letter and contain only lowercase letters, digits, and underscores."
    if (integrations_dir / name).exists():
        return f"Integration '{name}' already exists at integrations/{name}/."
    return None


def folder_to_package(folder_name: str) -> str:
    """Convert a folder name to a package name.

    amazon_bedrock -> amazon-bedrock-haystack
    """
    return folder_name.replace("_", "-") + "-haystack"


def folder_to_label(folder_name: str) -> str:
    """Convert a folder name to a GitHub label.

    amazon_bedrock -> integration:amazon-bedrock
    """
    return "integration:" + folder_name.replace("_", "-")


_SINGULAR_OVERRIDES: dict[str, str] = {
    "tracing": "tracing",
}


def singularize_type(component_type: str) -> str:
    """Return the singular form of a component type for use in module names.

    Falls back to stripping the trailing character, which works for regular
    plural forms (e.g. ``connectors`` -> ``connector``).
    """
    return _SINGULAR_OVERRIDES.get(component_type, component_type[:-1])


def get_module_path(folder_name: str, component_type: str) -> str:
    """Return the dotted import path for a given folder name and component type."""
    if component_type == "document_stores":
        return f"haystack_integrations.{component_type}.{folder_name}"
    return f"haystack_integrations.components.{component_type}.{folder_name}"
