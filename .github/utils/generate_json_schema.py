#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import List

import os
import sys
import json
import logging
from pathlib import Path

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

from haystack.nodes._json_schema import (
    find_subclasses_in_modules,
    create_schema_for_node_class,
)


logging.basicConfig(level=logging.INFO)


BRANCH_NAME = "text2speech"  # FIXME should be main after merge


def get_package_json_schema(
    title: str, description: str, module_names: List[str], schema_ref: str
):
    """
    Generate JSON schema for the custom node(s).
    """
    # List all known nodes in the given modules
    possible_node_classes = []
    for module_name in module_names:
        importlib.import_module(module_name)
        possible_node_classes += find_subclasses_in_modules(importable_modules=[module_name])

    # Build the definitions and refs for the nodes
    schema_definitions = []
    for _, node_class in possible_node_classes:
        node_definition, _ = create_schema_for_node_class(node_class)
        schema_definitions.append(node_definition)

    package_schema = {
        "$schema": "http://json-schema.org/draft-07/schema",
        "$id": schema_ref,
        "title": title,
        "description": f"{description} For more info read the docs at: https://haystack.deepset.ai/components/pipelines#yaml-file-definitions",
        "type": "object",
        "items": {"anyOf": schema_definitions},
    }
    return package_schema


def update_json_schema(
    destination_path: Path,
    version: str,
    title: str,
    description: str,
    package_name: str,
    module_names: str,
):
    """
    If the version contains "rc", only update main's schema.
    Otherwise, create (or update) a new schema.
    """
    base_schema_ref = f"https://raw.githubusercontent.com/deepset-ai/haystack-extras/{BRANCH_NAME}/nodes/{package_name}/json-schemas/"
    main_filename = f"haystack-{package_name}-main.schema.json"

    # Create the schemas for the nodes
    package_schema = get_package_json_schema(
        title=title,
        description=description,
        module_names=module_names,
        schema_ref=base_schema_ref + main_filename,
    )

    # Update mains's schema
    with open(destination_path / main_filename, "w+") as json_file:
        json.dump(package_schema, json_file, indent=2)

    # If it's not an rc version:
    if "rc" not in version:

        # Create/update the specific version file too
        version_filename = f"haystack-{package_name}-{version}.schema.json"
        package_schema["$id"] = base_schema_ref + version_filename
        with open(destination_path / version_filename, "w") as json_file:
            json.dump(package_schema, json_file, indent=2)

        # Update the index
        index_name = f"haystack-{package_name}.schema.json"

        if not os.path.exists(destination_path / index_name):
            generate_schema_index(
                destination_path=destination_path,
                schema_ref=base_schema_ref + index_name,
                title=title,
                description=description,
                index_name=index_name,
            )

        with open(destination_path / index_name, "r") as json_file:
            index = json.load(json_file)
            new_entry = {
                "allOf": [
                    # {"properties": {"version": {"const": version}}},  # FIXME once we agree on versioning, if necessary
                    {"$ref": base_schema_ref + version_filename},
                ]
            }
            if new_entry not in index["oneOf"]:
                index["oneOf"].append(new_entry)
        with open(destination_path / index_name, "w") as json_file:
            json.dump(index, json_file, indent=2)


def generate_schema_index(
    destination_path: Path,
    schema_ref: str,
    title: str,
    description: str,
    index_name: str,
):
    index = {
        "$schema": "http://json-schema.org/draft-07/schema",
        "$id": schema_ref,
        "title": title,
        "description": description,
        "type": "object",
        "oneOf": [],
    }
    with open(destination_path / index_name, "w") as json_file:
        json.dump(index, json_file, indent=2)


def get_package_data(folder: str):
    package_name = "haystack-" + folder
    meta = metadata.metadata(package_name)
    return {
        "package_name": package_name,
        "version": metadata.version("haystack-" + folder),
        "title": str(meta["name"]).replace("-", " ").replace("_", " "),
        "description": meta["summary"],
        "destination_path": (
            Path(sys.argv[0]).parent.parent.parent / "nodes" / folder / "json-schemas"
        ).absolute(),
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="JSON Schema generator for Haystack custom node packages"
    )
    parser.add_argument(
        "-f",
        "--folder-name",
        dest="folder",
        help="Name of the folder, i.e. hello-world-node",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--modules",
        dest="module_names",
        help="Name of the module, i.e. hello_world_node",
        required=True,
    )
    params = vars(parser.parse_args())
    package_data = get_package_data(folder=params["folder"])
    update_json_schema(**package_data, module_names=params["module_names"].split(","))
