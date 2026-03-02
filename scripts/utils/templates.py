# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Template strings for scaffolded integration files."""

import textwrap

from .naming import folder_to_label, folder_to_package, get_module_path, singularize_type


def workflow_yml(name: str) -> str:
    """Return the GitHub Actions workflow YAML for the integration."""
    return textwrap.dedent(f"""\
        # This workflow comes from https://github.com/ofek/hatch-mypyc
        # https://github.com/ofek/hatch-mypyc/blob/5a198c0ba8660494d02716cfc9d79ce4adfb1442/.github/workflows/test.yml
        name: Test / {name}

        on:
          schedule:
            - cron: "0 0 * * *"
          pull_request:
            paths:
              - "integrations/{name}/**"
              - "!integrations/{name}/*.md"
              - ".github/workflows/{name}.yml"

        defaults:
          run:
            working-directory: integrations/{name}

        concurrency:
          group: {name}-${{{{ github.head_ref }}}}
          cancel-in-progress: true

        env:
          PYTHONUNBUFFERED: "1"
          FORCE_COLOR: "1"

        jobs:
          run:
            name: Python ${{{{ matrix.python-version }}}} on ${{{{ startsWith(matrix.os, 'macos-') && 'macOS' || startsWith(matrix.os, 'windows-') && 'Windows' || 'Linux' }}}}
            runs-on: ${{{{ matrix.os }}}}
            strategy:
              fail-fast: false
              matrix:
                os: [ubuntu-latest, windows-latest, macos-latest]
                python-version: ["3.10", "3.13"]

            steps:
              - name: Support longpaths
                if: matrix.os == 'windows-latest'
                working-directory: .
                run: git config --system core.longpaths true

              - uses: actions/checkout@v6

              - name: Set up Python ${{{{ matrix.python-version }}}}
                uses: actions/setup-python@v6
                with:
                  python-version: ${{{{ matrix.python-version }}}}

              - name: Install Hatch
                run: pip install --upgrade hatch
              - name: Lint
                if: matrix.python-version == '3.10' && runner.os == 'Linux'
                run: hatch run fmt-check && hatch run test:types

              - name: Run tests
                run: hatch run test:cov-retry

              - name: Run unit tests with lowest direct dependencies
                run: |
                  hatch run uv pip compile pyproject.toml --resolution lowest-direct --output-file requirements_lowest_direct.txt
                  hatch -e test env run -- uv pip install -r requirements_lowest_direct.txt
                  hatch run test:unit

              - name: Nightly - run tests with Haystack main branch
                if: github.event_name == 'schedule'
                run: |
                  hatch env prune
                  hatch -e test env run -- uv pip install git+https://github.com/deepset-ai/haystack.git@main
                  hatch run test:cov-retry

              - name: Send event to Datadog for nightly failures
                if: failure() && github.event_name == 'schedule'
                uses: ./.github/actions/send_failure
                with:
                  title: |
                    Core integrations nightly tests failure: ${{{{ github.workflow }}}}
                  api-key: ${{{{ secrets.CORE_DATADOG_API_KEY }}}}
    """)


def pyproject_toml(name: str, component_type: str) -> str:
    """Return the pyproject.toml content for the integration."""
    pkg = folder_to_package(name)
    mod = get_module_path(name, component_type)
    return textwrap.dedent(f"""\
        [build-system]
        requires = ["hatchling", "hatch-vcs"]
        build-backend = "hatchling.build"

        [project]
        name = "{pkg}"
        dynamic = ["version"]
        description = "Haystack integration for {name}"
        readme = "README.md"
        requires-python = ">=3.10"
        license = "Apache-2.0"
        keywords = []
        authors = [{{ name = "deepset GmbH", email = "info@deepset.ai" }}]
        classifiers = [
          "License :: OSI Approved :: Apache Software License",
          "Development Status :: 4 - Beta",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Programming Language :: Python :: 3.12",
          "Programming Language :: Python :: 3.13",
          "Programming Language :: Python :: Implementation :: CPython",
          "Programming Language :: Python :: Implementation :: PyPy",
        ]
        dependencies = ["haystack-ai"]

        [project.urls]
        Documentation = "https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/{name}#readme"
        Issues = "https://github.com/deepset-ai/haystack-core-integrations/issues"
        Source = "https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/{name}"

        [tool.hatch.build.targets.wheel]
        packages = ["src/haystack_integrations"]

        [tool.hatch.version]
        source = "vcs"
        tag-pattern = 'integrations\\/{name}-v(?P<version>.*)'

        [tool.hatch.version.raw-options]
        root = "../.."
        git_describe_command = 'git describe --tags --match="integrations/{name}-v[0-9]*"'

        [tool.hatch.envs.default]
        installer = "uv"
        dependencies = ["haystack-pydoc-tools", "ruff"]

        [tool.hatch.envs.default.scripts]
        docs = ["haystack-pydoc pydoc/config_docusaurus.yml"]
        fmt = "ruff check --fix {{args}}; ruff format {{args}}"
        fmt-check = "ruff check {{args}} && ruff format --check {{args}}"

        [tool.hatch.envs.test]
        dependencies = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-rerunfailures",
            "mypy",
            "pip",
        ]

        [tool.hatch.envs.test.scripts]
        unit = 'pytest -m "not integration" {{args:tests}}'
        integration = 'pytest -m "integration" {{args:tests}}'
        all = 'pytest {{args:tests}}'
        cov-retry = 'pytest --cov=haystack_integrations --reruns 3 --reruns-delay 30 -x {{args:tests}}'
        types = "mypy -p {mod} {{args}}"

        [tool.mypy]
        install_types = true
        non_interactive = true
        check_untyped_defs = true
        disallow_incomplete_defs = true

        [tool.ruff]
        line-length = 120

        [tool.ruff.lint]
        select = [
            "A",
            "ARG",
            "B",
            "C",
            "DTZ",
            "E",
            "EM",
            "F",
            "I",
            "ICN",
            "ISC",
            "N",
            "PLC",
            "PLE",
            "PLR",
            "PLW",
            "Q",
            "RUF",
            "S",
            "T",
            "TID",
            "UP",
            "W",
            "YTT",
        ]
        ignore = [
            # Allow non-abstract empty methods in abstract base classes
            "B027",
            # Allow function calls in argument defaults (common Haystack pattern for Secret.from_env_var)
            "B008",
            # zip() without strict= is OK when lengths are guaranteed equal
            "B905",
            # Ignore checks for possible passwords
            "S105",
            "S106",
            "S107",
            # Ignore complexity
            "C901",
            "PLR0911",
            "PLR0912",
            "PLR0913",
            "PLR0915",
        ]
        unfixable = [
            # Don't touch unused imports
            "F401",
        ]

        [tool.ruff.lint.isort]
        known-first-party = ["haystack_integrations"]

        [tool.ruff.lint.flake8-tidy-imports]
        ban-relative-imports = "parents"

        [tool.ruff.lint.per-file-ignores]
        # Tests can use magic values, assertions, and relative imports
        "tests/**/*" = ["PLR2004", "S101", "TID252"]

        [tool.coverage.run]
        source = ["haystack_integrations"]
        branch = true
        parallel = false

        [tool.coverage.report]
        omit = ["*/tests/*", "*/__init__.py"]
        show_missing = true
        exclude_lines = ["no cov", "if __name__ == .__main__.:", "if TYPE_CHECKING:"]

        [tool.pytest.ini_options]
        addopts = "--strict-markers"
        markers = [
          "integration: integration tests",
        ]
        log_cli = true
        asyncio_default_fixture_loop_scope = "function"
    """)


def readme_md(name: str) -> str:
    """Return the README.md content for the integration."""
    pkg = folder_to_package(name)
    return textwrap.dedent(f"""\
        # {pkg}

        [![PyPI - Version](https://img.shields.io/pypi/v/{pkg}.svg)](https://pypi.org/project/{pkg})
        [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/{pkg}.svg)](https://pypi.org/project/{pkg})

        - [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/{name}/CHANGELOG.md)

        ---

        ## Contributing

        Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
    """)


def pydoc_config(name: str, component_type: str) -> str:
    """Return the pydoc docusaurus config YAML for the integration."""
    mod = get_module_path(name, component_type)
    title = name.replace("_", " ").title()
    return textwrap.dedent(f"""\
        loaders:
          - modules:
              - {mod}.{singularize_type(component_type)}
            search_path: [../src]
        processors:
          - type: filter
            documented_only: true
            skip_empty_modules: true
        renderer:
          description: {title} integration for Haystack
          id: integrations-{name}
          filename: {name.replace("_", "-")}.md
          title: {title}
    """)


def labeler_entry(name: str) -> str:
    """Return the labeler.yml block for the integration."""
    label = folder_to_label(name)
    return textwrap.dedent(f"""\
        {label}:
          - changed-files:
              - any-glob-to-any-file: "integrations/{name}/**/*"
              - any-glob-to-any-file: ".github/workflows/{name}.yml"
    """)


def readme_table_row(name: str, component_type: str, type_labels: dict[str, str]) -> str:
    """Return the markdown table row for the root README inventory."""
    pkg = folder_to_package(name)
    type_label = type_labels.get(component_type, component_type.replace("_", " ").title())
    return (
        f"| [{pkg}](integrations/{name}/)"
        f" | {type_label}"
        f" | [![PyPI - Version](https://img.shields.io/pypi/v/{pkg}.svg)](https://pypi.org/project/{pkg})"
        f" | [![Test / {name}](https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/{name}.yml/badge.svg)]"
        f"(https://github.com/deepset-ai/haystack-core-integrations/actions/workflows/{name}.yml)"
        f" |"
    )
