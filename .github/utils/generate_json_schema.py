#!/usr/bin/env python3

import importlib
from typing import List

import os
import sys
import json
import logging
from pathlib import Path
from importlib import metadata

from haystack.nodes._json_schema import find_subclasses_in_modules, create_schema_for_node_class


logging.basicConfig(level=logging.INFO)


def get_package_json_schema(title: str, description: str, module_name: str, schema_ref: str):
    """
    Generate JSON schema for the custom node.
    """
    # List all known nodes in the given modules
    importlib.import_module(module_name)
    possible_node_classes = find_subclasses_in_modules(importable_modules=[module_name])

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
    package_name: str = "text2speech-nodes",
    module_name: str = "text2speech_nodes",
):
    """
    If the version contains "rc", only update main's schema.
    Otherwise, create (or update) a new schema.
    """
    base_schema_ref = f"https://raw.githubusercontent.com/deepset-ai/haystack-extras/main/nodes/{package_name}/json-schemas/"
    main_filename = f"haystack-{package_name}-main.schema.json"

    # Create the schemas for the nodes
    package_schema = get_package_json_schema(
        title=title,
        description=description,
        module_name=module_name,
        schema_ref=base_schema_ref + main_filename
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
                    {"properties": {"version": {"const": version}}},
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
        "oneOf": []
    }
    with open(destination_path / index_name, "w") as json_file:
        json.dump(index, json_file, indent=2)



def get_package_data(folder: str):
    meta = metadata.metadata("haystack-"+folder)
    return {
        "version": metadata.version("haystack-"+folder),
        "title": str(meta["name"]).replace("-", " ").replace("_", " "),
        "description": meta["summary"],
        "destination_path": (Path(sys.argv[0]).parent.parent.parent / "nodes" / folder / "json-schemas").absolute()
    }



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='JSON Schema generator for Haystack custom node packages')
    parser.add_argument('-f','--folder-name', dest="folder", help='Name of the folder, i.e. hello-world-node', required=True)
    parser.add_argument('-m','--module', dest="module_name", help='Name of the module, i.e. hello_world_node', required=True)
    parser.add_argument('-v','--version', dest="version", help='Package version')
    parser.add_argument('-t','--title', dest="title", help='Schema title, i.e. "My Haystack Hello World Node"')
    parser.add_argument('-d','--description', dest="description", help='Schema description, i.e. "JSON schemas for Haystack nodes that can be used to greet the world."')
    parser.add_argument('-o','--output-path', dest="destination_path", help='Path where to save the generated schemas (usually <your package>/json-schemas)')
    params = vars(parser.parse_args())

    package_data = get_package_data(folder=params["folder"])

    update_json_schema(**package_data, module_name=params["module_name"])
