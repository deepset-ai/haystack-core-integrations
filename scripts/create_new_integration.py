# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Scaffold a new integration with all required boilerplate."""

import argparse
import sys
from pathlib import Path

from utils.naming import folder_to_package, get_module_path, validate_name
from utils.scaffold import (
    create_integration_files,
    create_workflow,
    update_coverage_comment_workflow,
    update_labeler,
    update_root_readme,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = REPO_ROOT / "integrations"

COMPONENT_TYPES = [
    "connectors",
    "converters",
    "document_stores",
    "downloaders",
    "embedders",
    "evaluators",
    "fetchers",
    "generators",
    "preprocessors",
    "rankers",
    "retrievers",
    "tools",
    "tracing",
    "translators",
]

TYPE_LABELS = {
    "connectors": "Connector",
    "converters": "Converter",
    "document_stores": "Document Store",
    "downloaders": "Downloader",
    "embedders": "Embedder",
    "evaluators": "Evaluator",
    "fetchers": "Fetcher",
    "generators": "Generator",
    "preprocessors": "Preprocessor",
    "rankers": "Ranker",
    "retrievers": "Retriever",
    "tools": "Tool",
    "tracing": "Tracer",
    "translators": "Translator",
}

LICENSE_HEADER = """\
# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""


def prompt_name() -> str:
    while True:
        name = input("Integration name (e.g. opensearch, amazon_bedrock): ").strip()
        error = validate_name(name, INTEGRATIONS_DIR)
        if error:
            print(f"  Error: {error}")
            continue
        return name


def prompt_component_type() -> str:
    print("Component type:")
    for i, t in enumerate(COMPONENT_TYPES, 1):
        print(f"  {i}. {t}")
    print(f"  {len(COMPONENT_TYPES) + 1}. (custom)")
    while True:
        choice = input(f"Choose [1-{len(COMPONENT_TYPES) + 1}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(COMPONENT_TYPES):
                return COMPONENT_TYPES[idx]
            if idx == len(COMPONENT_TYPES):
                custom = input("  Enter custom component type: ").strip().lower()
                if custom:
                    return custom
                print("  Component type cannot be empty.")
                continue
        except ValueError:
            if choice.lower() in COMPONENT_TYPES:
                return choice.lower()
        print(f"  Invalid choice. Enter a number between 1 and {len(COMPONENT_TYPES) + 1}.")


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new Haystack integration.")
    parser.add_argument("--name", type=str, help="The name of the integration (e.g. opensearch)")
    parser.add_argument("--type", dest="component_type", help=f"Component type: {', '.join(COMPONENT_TYPES)}")
    args = parser.parse_args()

    name = args.name
    if name is None:
        name = prompt_name()
    else:
        error = validate_name(name, INTEGRATIONS_DIR)
        if error:
            print(f"  Error: {error}", file=sys.stderr)
            sys.exit(1)

    component_type = args.component_type
    if component_type is None:
        component_type = prompt_component_type()
    elif component_type not in COMPONENT_TYPES:
        print(f"  Warning: '{component_type}' is not a known component type.")
        print(f"  Known types: {', '.join(COMPONENT_TYPES)}")
        confirm = input("  Continue anyway? [y/N]: ").strip().lower()
        if confirm != "y":
            sys.exit(1)

    package_name = folder_to_package(name)
    module_path = get_module_path(name, component_type)

    print(f"\nCreating integration: {package_name}")
    print(f"  Folder: integrations/{name}")
    print(f"  Module: {module_path}")

    created_files = create_integration_files(
        name, component_type, repo_root=REPO_ROOT, integrations_dir=INTEGRATIONS_DIR, license_header=LICENSE_HEADER
    )
    for file in created_files:
        print(f"  Created: {file.relative_to(REPO_ROOT)}")

    workflow_path = create_workflow(name, repo_root=REPO_ROOT)
    print(f"  Created: {workflow_path}")

    labeler_path = update_labeler(name, repo_root=REPO_ROOT)
    print(f"  Updated: {labeler_path}")

    coverage_path = update_coverage_comment_workflow(name, repo_root=REPO_ROOT)
    print(f"  Updated: {coverage_path}")

    readme_path = update_root_readme(name, component_type, TYPE_LABELS, repo_root=REPO_ROOT)
    print(f"  Updated: {readme_path}")

    print(f"\nDone! Next steps:")
    print(f"  1. Add your component code in integrations/{name}/src/")
    print(f"  2. Export the component classes from the __init__.py in your module")
    print(f"  3. Add integration-specific dependencies to integrations/{name}/pyproject.toml")
    print(f"  4. Add tests in integrations/{name}/tests/")
    print(f"  5. If your integration tests need API keys, update .github/workflows/{name}.yml accordingly")
    print(f"     and ask a maintainer to add the secret to the GitHub repo.")
    print(f"  6. Add relevant keywords to integrations/{name}/pyproject.toml")
    print(f"  7. Check that the correct module paths are used in integrations/{name}/pydoc/config_docusaurus.yml")


if __name__ == "__main__":
    main()
