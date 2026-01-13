import re
import requests

# * integrations/<INTEGRATION_FOLDER_NAME>-v1.0.0
# * integrations/<INTEGRATION_FOLDER_NAME>-v1.0.0.post0 (for post-releases)
INTEGRATION_VERSION_REGEX = r"integrations/([a-zA-Z_]+)-v([0-9]+\.[0-9]+\.[0-9]+(?:\.post[0-9]+)?)"


def validate_version_number(tag: str):
    """
    Verify that a release version number follows semantic versioning rules by checking the latest version on PyPI.
    Supports both regular versions (x.y.z) and post-release versions (x.y.z.postN) per PEP 440.
    """

    matches = re.match(INTEGRATION_VERSION_REGEX, tag)
    if not matches or len(matches.groups()) != 2:
        raise ValueError(f"Invalid tag: {tag}")

    integration_name, version_to_release = matches.groups()
    print(f"Integration name: {integration_name}")
    print(f"Integration version to release: {version_to_release}")

    # Replace underscores with hyphens to look for the package on PyPi
    integration_package = f"{integration_name.replace('_', '-')}-haystack"
    print(f"Integration PyPi package: {integration_package}")

    # connect to PyPi and check the latest version
    try:
        response = requests.get(f"https://pypi.org/pypi/{integration_package}/json")
        response.raise_for_status()
        latest_version = response.json()["info"]["version"]
        print(f"Latest version on PyPI: {latest_version}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404 and version_to_release in ["0.0.1", "0.1.0", "1.0.0"]:
            print("Package not found on PyPI. Assuming this is the first release and skipping version check.")
            return
        raise e

    # Check if this is a post-release version
    if ".post" in version_to_release:
        # For post-releases (e.g., 5.4.0.post0), the base version should match the latest on PyPI
        base_version = version_to_release.split(".post")[0]
        if base_version == latest_version:
            print(f"Post-release version {version_to_release} is valid for base version {base_version}")
            return
        else:
            msg = (
                f"Invalid post-release version: {version_to_release}. "
                f"Post-release must be based on the latest PyPI version: {latest_version}"
            )
            raise ValueError(msg)

    # Regular version check
    x, y, z = [int(i) for i in latest_version.split(".")]

    acceptable_new_versions = [
        f"{x}.{y}.{z + 1}",
        f"{x}.{y + 1}.0",
        f"{x + 1}.0.0",
    ]
    if version_to_release not in acceptable_new_versions:
        msg = (
            f"Invalid version to release: {version_to_release}. Acceptable new versions are: {acceptable_new_versions}"
        )
        raise ValueError(msg)
    print("The version to release is acceptable.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", help="Git tag in format 'integrations/<name>-v<version>'", required=True, type=str)
    args = parser.parse_args()

    validate_version_number(args.tag)
