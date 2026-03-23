# Haystack Core Integrations — Guidelines for AI Agents

## Repo Structure

This is a **monorepo** of independent Haystack integrations. Each integration lives under `integrations/<name>/` and 
is a self-contained Python package published to PyPI. 
Each integration is a namespace package under the `haystack_integrations` namespace.

Unless you are working on repository-level changes, you should `cd` into the target integration directory:

cd integrations/<integration_name>

All `hatch` commands below must be run from inside an integration directory, not from the repo root.

## Environment

Haystack Core Integrations uses **Hatch** for environment and dependency management.

Do not run `python` or `pip` directly.

Before running code on this project, you must be able to run `hatch --version` and get a correct output.

If not, ask the user where Hatch is or if they want to install it. For installation instructions, 
refer to https://hatch.pypa.io/latest/install/#installation.

### Run scripts

hatch run python SCRIPT.py

### Open a shell with installed dependencies

hatch shell

### Install temporary dependencies (for experiments only)

uv pip install PACKAGE

### Delete the environments

hatch env prune

## Tests

Tests run via Hatch and support pytest arguments.

### Run unit tests

hatch run test:unit

### Run integration tests

hatch run test:integration

Some integrations require API keys or running containers for integration tests. 
Check the integration's README for specific instructions.

## Quality Checks

### Type checking with mypy
hatch run test:types

To fix type issues, avoid `type: ignore`, casts, or assertions when possible. If they are necessary, explain why.

### Format and lint
hatch run fmt

## Versioning

Each integration is versioned independently via git tags with the pattern `integrations/<name>-v<version>` (e.g. `integrations/anthropic-v5.7.0`).

Only maintainers can release new versions of integrations, following the instructions in the general `README.md`.

## Changelogs

Changelogs are auto-generated per integration and not meant to be edited manually.

## Creating a New Integration

Follow the instructions in the "Create a new integration" section of `CONTRIBUTING.md`.
