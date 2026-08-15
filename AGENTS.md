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

<!-- braindump:begin -->

The rules below were mined from 2,772 PR review comments written by the deepset
team between 2025-07-01 and 2026-08-14, then filtered against the current source tree so that
guidance referring to APIs removed or moved in Haystack 3.0 does not survive.

They describe what reviewers actually enforce. Follow them the way you would follow a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

Also see directory-specific guidelines:

- [integrations/AGENTS.md](integrations/AGENTS.md)

## API Design

- Target Haystack 3.x APIs: use `ChatGenerator`s, create clients in `warm_up()`, and support the module allowlist — Keeps integrations compatible with Haystack 3.x lifecycle, chat APIs, and safe deserialization behavior.
- Keep `to_dict()`/`from_dict()` symmetric with `__init__` — preserve all runtime config, including Watsonx `max_retries`, `meta_fields_to_embed`, and `embedding_separator`.
- Keep Haystack `__init__` light; route reusable setup through `warm_up()` — prevents slow construction and duplicated lifecycle logic, especially in `integrations/mcp/src/haystack_integrations/tools/mcp/`
- Preserve protocol parameter order; append store-specific args and pass ambiguous optionals by keyword — Maintains protocol compatibility while allowing implementations to add store-specific options without breaking callers.
- Align public API names across signatures, docs, code, and returns — document intentional mismatches
- Make required `run(...)` inputs explicit keyword-only args — prevents no-op component runs
- Use `StreamingChunk.reasoning`/`ReasoningContent` for reasoning — keep `meta` incidental
- Resolve `Secret` values in `warm_up`, not `__init__` — avoids init-time side effects
- Use Haystack serialization shape with runtime `type` — e.g. `{"type": generate_qualified_class_name(type(self)), "data": ...}`
- Align `Document` embedder constructors in `haystack/components/embedders` and `integrations/*/src/haystack_integrations/components/embedders/*` — include provider-relevant options like `meta_fields_to_embed`, `embedding_separator`, `prefix`, `suffix`, and `batch_size` when comparable embedders expose them
- Make `warm_up()` idempotent with an `__init__` flag set only after setup succeeds — This prevents repeated expensive setup and avoids partially marking failed initialization as ready.
- Type chat generator `tools` as `ToolsType`; don’t narrow provider-native params — preserves Haystack API compatibility and tool pass-through

## Documentation

- Keep comments concise and substantive — explain non-obvious intent, limits, edge cases, workarounds, or real compat needs
- Update `pydoc/config_docusaurus.yml` for public API changes — include modules, retrievers, and errors
- Keep docstrings authoritative and preserve reference links — prevents stale or duplicated docs
- Add the standard `SPDX-FileCopyrightText` and `Apache-2.0` header before imports — Ensures repository-wide license compliance and keeps source and test files consistent across core and integration packages.
- Keep public examples current with supported APIs — update docstrings, cookbooks, integrations, Google GenAI model names, and `RagasEvaluator` `ragas.metrics.collections` usage

## Config

- Source workflow creds from matching `${{ secrets.<SECRET_NAME> }}` env vars — keeps CI secure and reliable
- Align workflow `python-version` matrices to min/max supported Python — catches boundary regressions
- Default secrets from provider env vars; use `WATSONX_API_KEY` for Watsonx components — Predictable env-var defaults make credentials easy to configure across integrations while avoiding hardcoded secrets or setup surprises.
- Use SDK env var names consistently across CI, tests, secrets, and skips — prevents config drift
- Store credential fields as Haystack `Secret`, not `str` — avoids leaking sensitive config

## Testing

- Gate `integrations/<provider>/tests/` explicitly — wire CI coverage, inject `Secret`/env values, use provider-specific `pytest.mark.skipif(...)` reasons, precise `sys.version_info`/`sys.platform` checks, and validate external-API tests locally with a personal key.
- Test each advertised format variant, including optional params like `embedding_types` — catches non-default response and content bugs
- Include all supported OSes in `.github/workflows/` test matrices — catches OS-specific bugs

## Code Style

- Update Haystack `Document`s with `dataclasses.replace(...)` — avoids in-place mutation bugs
- Exclude generated artifacts from feature PRs — let release/merge workflows create them
- Inline tiny private helpers used once — keeps control flow clear and avoids indirection

## Dependencies

- Run `hatch` inside `integrations/<name>/` — each integration owns its envs
- Pin `.github/workflows/` Action `uses:` deps to full commit SHAs — avoid mutable refs

## Imports

- Import moved 3.0 integrations from `haystack_integrations`, not `haystack` — avoids broken imports
- Import `from haystack import logging` for project logging; keep `logging.getLogger(__name__)` — This uses Haystack’s logging behavior consistently across tests, components, and integrations while preserving standard logger naming.

## General

- Type inputs/returns precisely; replace `Any` when supported values or shapes are known — Precise structures and element types improve type checking, document API contracts, and prevent shape-related bugs.
- Use `metadata_field`/`metadata_fields` in public APIs — clarifies document metadata args
- Raise `TypeError` for wrong input types/shapes; `ValueError` for invalid values/config — This preserves consistent API misuse semantics and helps callers distinguish type errors from invalid values.

## Topic Guides

Check these when working in specific areas:

- **[API Docstring Style](agent_docs/api-docstring-style.md)**: When writing or updating public API, component, constructor, method, or usage-example docstrings

## File-Specific Rules

### `README.md`

- Sort the `README.md` integrations table alphabetically — keeps entries findable and diffs clean

<!-- braindump:end -->
