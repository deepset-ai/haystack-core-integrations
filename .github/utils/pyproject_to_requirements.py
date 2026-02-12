import argparse
from pathlib import Path
import toml

def main(pyproject_path: Path, exclude_optional_dependencies: bool = False):
    content = toml.load(pyproject_path)
    deps = set(content["project"]["dependencies"])

    if not exclude_optional_dependencies:
        optional_deps = content["project"].get("optional-dependencies", {})
        for dep_list in optional_deps.values():
            deps.update(dep_list)

    print("\n".join(sorted(deps)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pyproject_to_requirements.py",
        description="Convert pyproject.toml to requirements.txt"
    )
    parser.add_argument("pyproject_path", type=Path, help="Path to pyproject.toml file")
    parser.add_argument("--exclude-optional-dependencies", action="store_true", help="Exclude optional dependencies")
    
    args = parser.parse_args()
    main(args.pyproject_path, args.exclude_optional_dependencies)
