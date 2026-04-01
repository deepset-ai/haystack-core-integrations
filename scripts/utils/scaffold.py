# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""File creation and config-file update operations."""

import re
from pathlib import Path
from string import Template

from .naming import folder_to_label, folder_to_package, get_module_path, singularize_type


_TEMPLATES_DIR = Path(__file__).parent / "templates"


def render(filename: str, **kwargs: str) -> str:
    """Load a template from the templates/ directory and substitute variables."""
    text = (_TEMPLATES_DIR / filename).read_text()
    return Template(text).safe_substitute(**kwargs)


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

    pkg = folder_to_package(name)
    mod = get_module_path(name, component_type)

    write_file("pyproject.toml", render("pyproject.toml", name=name, pkg=pkg, mod=mod))
    write_file("README.md", render("readme.md", name=name, pkg=pkg))
    license_src = repo_root / "LICENSE"
    write_file("LICENSE.txt", license_src.read_text())
    write_file(
        "pydoc/config_docusaurus.yml",
        render(
            "pydoc_config.yml",
            name=name,
            mod=mod,
            title=name.replace("_", " ").title(),
            singular_type=singularize_type(component_type),
            name_hyphenated=name.replace("_", "-"),
        ),
    )

    return created_files


def create_workflow(name: str, *, repo_root: Path) -> str:
    """Create a GitHub Actions workflow file. Returns the relative path."""
    workflow_path = repo_root / ".github" / "workflows" / f"{name}.yml"
    workflow_path.write_text(render("workflow.yml", name=name))
    return str(workflow_path.relative_to(repo_root))


def update_labeler(name: str, *, repo_root: Path) -> str:
    """Insert a labeler.yml entry in alphabetical order. Returns the relative path."""
    labeler_path = repo_root / ".github" / "labeler.yml"
    content = labeler_path.read_text()
    new_label = folder_to_label(name)
    entry = render("labeler_entry.yml", name=name, label=new_label)

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


def update_coverage_comment_workflow(name: str, *, repo_root: Path) -> str:
    """Insert a workflow trigger entry in CI_coverage_comment.yml in alphabetical order. Returns the relative path."""
    workflow_path = repo_root / ".github" / "workflows" / "CI_coverage_comment.yml"
    content = workflow_path.read_text()

    new_entry = f'      - "Test / {name}"'

    # Find all existing workflow trigger entries and insert in sorted order
    entry_pattern = re.compile(r'^      - "Test / ([^"]+)"', re.MULTILINE)
    matches = list(entry_pattern.finditer(content))

    insert_pos = None
    for match in matches:
        if match.group(1) > name:
            insert_pos = match.start()
            break

    if insert_pos is not None:
        content = content[:insert_pos] + new_entry + "\n" + content[insert_pos:]
    else:
        # Insert after the last entry (before the `types:` line)
        last_match = matches[-1]
        insert_pos = last_match.end()
        content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]

    workflow_path.write_text(content)
    return str(workflow_path.relative_to(repo_root))


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

    new_pkg = folder_to_package(name)
    type_label = type_labels.get(component_type, component_type.replace("_", " ").title())
    row = render("readme_table_row.txt", name=name, pkg=new_pkg, type_label=type_label).rstrip("\n")

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
