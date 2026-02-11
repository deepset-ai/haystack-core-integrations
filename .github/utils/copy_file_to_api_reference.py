"""
Script to copy a generated markdown file to the Haystack API reference (integrations-api).

Assumes that Haystack Docs Website is available in the working directory.
Copies the file to the main API reference and the versioned API references.

Usage:
python copy_file_to_api_reference.py <generated_markdown_file>
"""

import argparse
import os
import shutil

def copy_file_to_api_reference(generated_markdown_file: str):
    shutil.copy(generated_markdown_file, "docs-website/reference/integrations-api/")
    for version_dir in os.scandir("docs-website/reference_versioned_docs"):
        if version_dir.is_dir():
            # example: docs-website/reference_versioned_docs/version-2.17/integrations-api
            integrations_api_ref_dir = os.path.join(version_dir.path, "integrations-api")
            shutil.copy(generated_markdown_file, integrations_api_ref_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_markdown_file", type=str, help="Path to the generated markdown file")
    args = parser.parse_args()
    copy_file_to_api_reference(args.generated_markdown_file)