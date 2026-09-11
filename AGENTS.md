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

In addition, there are rules inferred from previous code reviews. Follow them like a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

## API Design

- Target Haystack 3.x: use ChatGenerators, create clients in `warm_up()`, support the deserialization module allowlist, and import components that moved out of core from `haystack_integrations`, not `haystack`
- Keep `to_dict()`/`from_dict()` symmetric with `__init__` — every constructor argument that affects runtime behaviour round-trips (retry, batching, metadata options); use `default_to_dict`/`default_from_dict` whenever `init_parameters` can rebuild the component, and the `{"type": generate_qualified_class_name(type(self)), "data": ...}` shape only when custom serialization is unavoidable
- Keep `__init__` light: create clients, resolve `Secret`s, and turn `None` attributes into instances in `warm_up()`, made idempotent by a flag set only after setup succeeds; validate concrete backend objects in `__init__` so incompatible dependencies fail fast
- Keep public signatures protocol-compatible: preserve protocol parameter order and append store-specific args, make optional params keyword-only and required `run(...)` inputs explicit keyword-only, and keep Document Store APIs consistent across backends — callers can switch backends without breaking
- Expose only wired, supported params, with names aligned across signatures, docs, and returns; reject or document unsupported filters/flags and any intentional mismatch
- Stream through `streaming_callback`, never by returning chunks in outputs: preserve provider-native stream indices, set `ToolCallDelta.index` from provider-stable call IDs, carry reasoning in `StreamingChunk.reasoning`/`ReasoningContent` rather than `meta`, and handle unsupported chunk shapes explicitly
- Align `Document` embedder constructors with core and sibling integrations — include `meta_fields_to_embed`, `embedding_separator`, `prefix`, `suffix`, and `batch_size` when comparable embedders expose them
- Type chat generator `tools` as `ToolsType` and pass provider-native params through unnarrowed; declare `SUPPORTED_MODELS` beside components that support only a model subset
- Match metadata/search substrings case-insensitively as literals, not `LIKE` patterns — prevents `%`/`_` wildcard bugs
- Store credentials as `Secret`, never `str`, defaulting each from the provider's conventional env var (`COHERE_API_KEY`, `NVIDIA_API_KEY`, `WATSONX_API_KEY`, ...); use the SDK's exact env var names consistently across code, CI, tests, and skips
- Add async APIs only for native async I/O; keep sync and async methods symmetrical — shared filter/count/arg/error helpers, mirrored contracts, gates, fixtures, and assertions in tests
- Converters: accept `meta` as `dict | list[dict] | None` for multi-`sources`, merge `ByteStream.meta` into `Document.meta`, and keep original paths (`ByteStream.meta["file_path"]`) as provenance instead of temp filenames
- Document stores: use backend-native bulk APIs, apply filters before iterating, aggregating, or paginating (reuse `filter_documents(filters=filters)`), and wrap backend failures as `DocumentStoreError` consistently in sync and async paths, preserving documented bulk write/delete error behaviour
- Expose reusable functionality as named importable APIs — e.g. a `@component` plus its tool wrapper (`GitHubFileEditor`, `GitHubFileEditorTool`)
- Use `filter_policy` with `apply_filter_policy(...)` for retriever filters — avoids inconsistent merges

## Documentation

- Keep comments concise and substantive, in tests too — non-obvious intent, limits, edge cases, workarounds, real compat needs
- Update `pydoc/config_docusaurus.yml` when public modules, retrievers, or errors change, but leave generated artifacts (`CHANGELOG.md`, API docs) to the release and merge workflows
- Write Haystack-style docstrings — one-line summary, blank line, unindented sections; document every public param by purpose and constraints (defaults live in the signature; `__init__` params in the `__init__` docstring), `:returns:` with mapping keys/types matching `@component.output_types(...)`, concrete `:raises:` conditions (no `If ...` placeholders), and performance caveats such as slow queries; keep async variants in sync and preserve reference links
- Start every source and test file with the `SPDX-FileCopyrightText`/`Apache-2.0` header, and fill `integrations/*/LICENSE.txt` copyright years and holders with real values
- Keep public examples and docs current with supported APIs — refresh docstrings, cookbooks, and integration docs when model names or provider APIs change; document model-dependent behavior with canonical provider links and say whether model lists are exhaustive
- In docstrings use single backticks for inline code (double backticks only in release notes), single-line Markdown links, and `### Usage example` with fenced ` ```python ` blocks — avoids renderer issues
- Keep `integrations/*/README.md` minimal and template-aligned — link to canonical docs instead of duplicating them, list only the prerequisites local tests need, and delete obsolete docs, examples, deps, and configs
- Document the real contract: `None` fallbacks and env/upstream defaults, option precedence when maps or headers override named options, auth modes by credential presence, value ranges and external limits, and actual retrieval/scoring support
- Give context-complete examples — required `Document` setup, execution, and produced output
- Sort the `README.md` integrations table alphabetically — keeps entries findable and diffs clean

## Testing

- Gate `integrations/<provider>/tests/` explicitly — wire CI coverage, inject `Secret`/env values, give provider-specific `pytest.mark.skipif(...)` reasons with precise `sys.version_info`/`sys.platform` checks, and run external-API tests locally with a personal key first
- Cover every advertised format variant (optional params like `embedding_types`) and converter skip/failure paths with their warnings — non-default paths regress silently
- Organize tests by component — document-store coverage in `test_document_store.py` and `test_document_store_async.py`, fixtures in `tests/test_files/`, no one-off files
- Keep tests lean: parametrize duplicates with `@pytest.mark.parametrize`, use local fixtures only for shared non-trivial setup, and delete unused fixtures, helpers, and setup
- Base document-store tests on the `haystack.testing.document_store` classes, assert persisted state after mutating operations, and cover `close()`/reopen lifecycle changes
- Test `to_dict()`/`from_dict()` round trips with non-default init params, including that `Secret`-backed credentials restore and resolve; test only real legacy serialization formats, not invented shims
- In `integrations/*/tests/test_*chat_generator*.py` test integration-layer behavior only (core suites cover the rest): helper conversions including provider reasoning/thinking content, same-turn multi-tool calls modeled in one assistant message, and mixed init/runtime tools asserting merged `config.tools`, mirrored sync/async
- In init tests pass explicit non-default args (`model`, backend IDs) and call `warm_up()` only when asserting warm-up behavior
- Test provider streaming end-to-end from realistic chunk sequences — every `StreamingChunk`, metadata/usage/finish fields, tool-call/reasoning output, and the final `ChatMessage`

## Type System

- Prefer type-correct code over `# type: ignore`; when unavoidable, use an exact-line `# type: ignore[code]` with a comment on why it is safe
- Place `py.typed` at the exposed package boundary (e.g. `haystack_integrations/tools/py.typed`) — enables type discovery across provider subdirectories
- Type precisely: replace `Any` when shapes are known, align `run()` return annotations with `@component.output_types(...)` and the returned `dict[...]`, and use direct annotations instead of quoted strings unless a real forward reference needs them

## Code Style

- Update `Document`s with `dataclasses.replace(...)`, never in place — shared documents leak mutations
- Inline tiny once-used private helpers; mark state-free helper methods `@staticmethod`
- Log with `from haystack import logging` and `logging.getLogger(__name__)`, using `{placeholder}` templates with kwargs — keeps logs structured
- Keep `haystack_integrations` roots and intermediate directories namespace-only (no `__init__.py`); in the concrete package's `__init__.py` re-export only the intentional public API
- Use canonical integration names everywhere (package dirs under `integrations/*`, READMEs, URLs, tables); prefix non-public helpers with `_`; name async counterparts `<method>_async`

## Config

- In `.github/workflows/`: source credentials from matching `${{ secrets.<NAME> }}` env vars, test the min and max supported Python versions on every supported OS, and pin `uses:` to full commit SHAs
- Align each `integrations/*/pyproject.toml` with `scripts/utils/templates/pyproject.toml`: real `authors`, approved `description`, `keywords`, Python metadata matching tested versions, and a docs env running `haystack-pydoc pydoc/config_docusaurus.yml` with only docs/lint deps
- Keep tooling in the same `pyproject.toml`: shared Ruff config with test-only ignores under `[tool.ruff.lint.per-file-ignores]` for `"tests/**/*"`, only used pytest markers with `--strict-markers`, mypy overrides for untyped imports instead of scattered `# type: ignore[import-untyped]`, and every importable package in the `types` env

## Dependencies

- Declare only directly used runtime deps with explicit minimums and no pins or upper bounds unless required; keep test deps minimal, relying on inherited ones
- Prefer the official provider SDK when it covers the workflow; for raw HTTP use `request_with_retry`/`async_request_with_retry` and expose `timeout`/`max_retries`

## General

- Use `metadata_field`/`metadata_fields` in public APIs — clarifies document metadata args
- Raise `TypeError` for wrong input types or shapes and `ValueError` for invalid values or config
- Import required deps at module top; reserve lazy or `try` imports for optional deps and import cycles — missing packages should fail at import
