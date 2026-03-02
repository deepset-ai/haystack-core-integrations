# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""File creation and config-file update operations."""

import re
from pathlib import Path

from .naming import folder_to_label, folder_to_package
from .templates import (
    labeler_entry,
    pydoc_config,
    pyproject_toml,
    readme_md,
    readme_table_row,
    workflow_yml,
)


def create_integration_files(
    name: str,
    component_type: str,
    *,
    repo_root: Path,
    integrations_dir: Path,
    license_header: str,
) -> list[Path]:
    """Create all files inside integrations/{name}/."""
    created_files: list[Path] = []
    base_dir = integrations_dir / name
    base_dir.mkdir(parents=True, exist_ok=True)

    def write_file(relative_path: str, content: str):
        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        created_files.append(path)

    if component_type == "document_stores":
        src_dir = f"src/haystack_integrations/{component_type}"
    else:
        src_dir = f"src/haystack_integrations/components/{component_type}"

    write_file(f"{src_dir}/{name}/__init__.py", license_header)
    write_file(f"{src_dir}/py.typed", "")

    write_file("tests/__init__.py", license_header)

    write_file("pyproject.toml", pyproject_toml(name, component_type))
    write_file("README.md", readme_md(name))
    license_src = repo_root / "LICENSE"
    write_file("LICENSE.txt", license_src.read_text())
    write_file("pydoc/config_docusaurus.yml", pydoc_config(name, component_type))

    return created_files


def create_workflow(name: str, *, repo_root: Path) -> str:
    """Create a GitHub Actions workflow file. Returns the relative path."""
    workflow_path = repo_root / ".github" / "workflows" / f"{name}.yml"
    workflow_path.write_text(workflow_yml(name))
    return str(workflow_path.relative_to(repo_root))


def update_labeler(name: str, *, repo_root: Path) -> str:
    """Insert a labeler.yml entry in alphabetical order. Returns the relative path."""
    labeler_path = repo_root / ".github" / "labeler.yml"
    content = labeler_path.read_text()
    new_label = folder_to_label(name)
    entry = labeler_entry(name)

    label_pattern = re.compile(r"^(integration:\S+):", re.MULTILINE)
    matches = list(label_pattern.finditer(content))

    insert_pos = None
    for match in matches:
        if match.group(1) > new_label:
            insert_pos = match.start()
            break

    if insert_pos is not None:
        content = content[:insert_pos] + entry + "\n" + content[insert_pos:]
    else:
        content += entry

    labeler_path.write_text(content)
    return str(labeler_path.relative_to(repo_root))


def update_root_readme(
    name: str,
    component_type: str,
    type_labels: dict[str, str],
    *,
    repo_root: Path,
) -> str:
    """Insert a README table row in alphabetical order. Returns the relative path."""
    readme_path = repo_root / "README.md"
    content = readme_path.read_text()

    row = readme_table_row(name, component_type, type_labels)
    new_pkg = folder_to_package(name)

    row_pattern = re.compile(r"^\| \[([^\]]+)\]", re.MULTILINE)
    matches = list(row_pattern.finditer(content))

    insert_pos = None
    for match in matches:
        if match.group(1) > new_pkg:
            insert_pos = match.start()
            break

    if insert_pos is not None:
        content = content[:insert_pos] + row + "\n" + content[insert_pos:]
    else:
        marker = "\n## Releasing"
        if marker in content:
            content = content.replace(marker, row + "\n" + marker)
        else:
            content += f"\n{row}\n"

    readme_path.write_text(content)
    return str(readme_path.relative_to(repo_root))
