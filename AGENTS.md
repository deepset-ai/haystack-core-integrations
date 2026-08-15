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

The rules below were mined from 2,772 PR review comments written by the deepset
team between 2025-07-01 and 2026-08-14, then filtered against the current source tree so that
guidance referring to APIs removed or moved in Haystack 3.0 does not survive.

They describe what reviewers actually enforce. Follow them the way you would follow a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

## API Design

- Target Haystack 3.x APIs: use `ChatGenerator`s, create clients in `warm_up()`, and support the module allowlist — Keeps integrations compatible with Haystack 3.x lifecycle, chat APIs, and safe deserialization behavior.
- Keep `to_dict()`/`from_dict()` symmetric with `__init__` — every constructor argument that affects runtime behaviour must round-trip, including retry, batching, and metadata options
- Keep `__init__` light; route reusable setup through `warm_up()` — avoids slow construction and duplicated lifecycle logic
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
- Keep public examples current with supported APIs — refresh docstrings, cookbooks, and integration docs when model names or provider APIs change

## Config

- Source workflow creds from matching `${{ secrets.<SECRET_NAME> }}` env vars — keeps CI secure and reliable
- Align workflow `python-version` matrices to min/max supported Python — catches boundary regressions
- Default each `Secret` from the provider's conventional env var (`COHERE_API_KEY`, `NVIDIA_API_KEY`, `WATSONX_API_KEY`, ...) — predictable defaults keep credential setup consistent and avoid hardcoded secrets
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

## File-Specific Rules

### `README.md`

- Sort the `README.md` integrations table alphabetically — keeps entries findable and diffs clean

## API Docstring Style

_When to check: When writing or updating public API, component, constructor, method, or usage-example docstrings_

- Use single backticks for inline code in prose; reserve double backticks for Haystack release notes — Single-backtick inline code renders consistently across docstrings and docs, while preserving the Haystack release-note exception.
- Document all public callable params; put explicit `__init__` params in the `__init__` docstring — Keeps public API docs accurate and ensures generated documentation shows constructor args in the expected place.
- Document performance caveats in API docstrings — warn about slow queries or large scans
- Write concise public docstrings and sync async variants — document real contracts, not internals
- Use single-line Markdown links in docstrings/comments — keeps docs readable and links rendering
- Document public returns with Sphinx `:returns:` — include mapping keys/types and match `@component.output_types(...)` names
- Describe `__init__` params by purpose/constraints; omit defaults already in signatures — Function signatures already expose defaults, so repeating them in docstrings creates stale documentation when defaults change.
- Document public method exceptions with concrete `:raises:` conditions — no `If ...` placeholders
- Write Haystack-style docstrings — one-line summary, blank line, then unindented sections/examples
- Use `### Usage example` and fenced ` ```python ` blocks for Python examples — avoids renderer issues

## Integration and Component Documentation

_When to check: When writing or updating READMEs, docstrings, examples, authentication docs, parameter documentation, LICENSE files, or generated documentation artifacts_

- Keep `integrations/*/README.md` minimal and template-aligned — link to canonical docs/examples instead of duplicating long content
- Leave generated docs under `integrations/` untouched — pipelines own `CHANGELOG.md` and API docs
- Document auth by credential presence and `Secret` inputs — prevents unclear setup paths
- Document only needed local test prerequisites in `integrations/{integration}/README.md` — avoids misleading setup
- Document option precedence in `integrations` docs — list named options overridden by maps/headers
- Document model-dependent APIs with canonical links — clarify supported kwargs, models, defaults, modes, outputs, and whether lists are exhaustive
- Fill `integrations/*/LICENSE.txt` copyright fields — use real years and holders, not placeholders
- Document non-obvious public API requirements — include ranges, external limits, syntax, and examples
- Sync constructor docs, defaults, and `None` fallbacks — state env/base/upstream defaults
- Document integrations with context-complete examples — show required `Document` setup, workflow execution, and output production
- Align integration retriever docstrings with actual retrieval/scoring support — avoids misleading users

## Conventions for `integrations/`

### Testing

- Put async document-store tests in `test_document_store_async.py` — keeps sync/async coverage clear
- Parametrize duplicate `pytest` tests — merge same behavior into one test with `@pytest.mark.parametrize`
- Base document store tests on `haystack.testing.document_store` classes — avoids duplicate, incomplete contract tests
- Test `to_dict()`/`from_dict()` round trips with non-default init params — preserve serializable config
- Test both sync and async integration paths — mirror gates, fixtures, inputs, and assertions
- Store fixtures in `integrations/{integration}/tests/test_files/` — keeps tests local, stable, and isolated
- Consolidate `integrations/<provider>/tests/` by component — add document-store coverage to `test_document_store.py`, not one-off files
- Test only changed integration chat behavior in `integrations/*/tests/test_*chat_generator*.py` — avoids redundant live/core coverage
- Avoid routine `warm_up()` in `integrations/*/tests` init tests — call it only when asserting warm-up behavior
- Assert persisted outcomes after mutating document-store tests — catches cleanup/index regressions
- Delete unused test fixtures/helpers/setup — keeps integration tests focused and maintainable
- Test chat helper conversions directly — cover provider-specific reasoning/thinking content
- Align pytest markers in `integrations/*/pyproject.toml` — declare only used markers and set `--strict-markers`
- Test converter skip/failure paths and warnings in `integrations/*/tests/` — preserves graceful-failure behavior
- Test same-turn multi-tool calls in chat generator integrations — model them in one assistant message
- Keep test comments specific and current — preserves intent and prevents stale guidance
- Test mixed chat tools across init/runtime paths — assert merged `config.tools` and mirror sync/async coverage
- Pass explicit init args in `integrations` tests — include `model`/backend IDs to validate custom paths
- Test `close()`/reopen changes in `integrations/*/tests/test_document_store.py` — keeps lifecycle coverage consistent
- Use local `pytest` fixtures only for shared non-trivial setup — keeps tests clear and uncoupled
- Test provider streaming end-to-end in `integrations/*/tests/test_chat_generator.py` — assert every `StreamingChunk`, metadata/usage/finish field, tool-call/reasoning output, and final `ChatMessage` from realistic provider chunk sequences.
- Test only real legacy serialization formats — avoid fake shims for missing current fields
- Assert secret-backed credentials restore and resolve — prevents hidden credential bugs in tests

### API Design

- Use keyword-only args for optional public API params — preserves backward compatibility
- Use case-insensitive literal substrings for metadata/search filters; in IBM DB use `LOCATE(UPPER(?), UPPER(column)) > 0`, not `LIKE` — Case-normalized literal matching keeps search behavior consistent across integrations and prevents `%`/`_` wildcard bugs in IBM DB.
- Use `Secret` for sensitive integration config/API values — prevents leaked credentials
- Prefix non-public helpers in `integrations` with `_` — clarifies API boundaries
- Add async APIs only for native async I/O; mirror sync contracts and tests — Keeps async APIs non-blocking and consistent with sync behavior, preventing event-loop stalls and contract drift.
- Keep public Document Store APIs consistent across backends — preserves portability
- Expose only wired, supported public params; reject or document unsupported filters/flags — Avoids misleading no-op APIs, runtime surprises, and inconsistent integration behavior.
- Use Haystack default serialization when `init_parameters` can rebuild the component — avoids brittle custom `to_dict`/`from_dict`; require valid `init_parameters` for custom deserialization.
- Set `ToolCallDelta.index` from provider-stable call IDs — preserves chunk correlation
- Accept `meta` as `dict | list[dict] | None` for multi-`sources` converters — keeps integration metadata semantics consistent
- Merge `ByteStream.meta` into converter `Document.meta` and document it — preserves metadata
- Declare `SUPPORTED_MODELS` beside limited integration components — documents model limits
- Use backend-native bulk APIs in integration document stores — improves throughput and avoids race-prone per-document logic
- Preserve provider-native stream indices in `StreamingChunk` — don’t hardcode or reshape for helpers
- Apply filters before iterating, aggregating, or paginating docs — reuse `filter_documents(filters=filters)` to prevent query bugs
- Use `streaming_callback` for `StreamingChunk`s — don't return chunks in outputs; preserve metadata, handle unsupported chunk shapes explicitly, and test streaming/non-streaming paths
- Validate concrete backend deps in `__init__` — fail fast when objects are incompatible
- Preserve converter source provenance — use original paths or `ByteStream.meta['file_path']`, not synthetic temp filenames
- Expose reusable integrations as named importable APIs — e.g., `@component` `GitHubFileEditor` plus `GitHubFileEditorTool`
- Use `filter_policy` and `apply_filter_policy(...)` for retriever filters — avoids inconsistent merge bugs

### Config

- Align `integrations/*/pyproject.toml` Hatch docs config — use `haystack-pydoc pydoc/config_docusaurus.yml` and only docs/lint deps like `haystack-pydoc-tools` and `ruff`
- Populate `project.keywords` in each `integrations/*/pyproject.toml` — improves package discoverability
- Align `integrations/*/pyproject.toml` Python metadata with tested support — prevents invalid installs and stale compatibility claims
- Align `integrations/<component>/pyproject.toml` Ruff config with the shared template; put test-only ignores under `[tool.ruff.lint.per-file-ignores]` for `"tests/**/*"` — keeps integration linting consistent without weakening global rules
- Align `integrations/*/pyproject.toml` with the canonical template — ensures consistent packaging
- Set real `[project].authors` in `integrations/*/pyproject.toml` — captures true maintainer and partner ownership
- Set `description` in `integrations/*/pyproject.toml` to approved integration wording — avoids vague package metadata

### Type System

- Prefer type-correct code over `# type: ignore`; if needed, use exact-line `# type: ignore[code]` with a safety comment — Type-correct code avoids hiding real bugs, while targeted suppressions keep unavoidable checker limitations auditable and safe.
- Place integration `py.typed` at the exposed package boundary, e.g. `haystack_integrations/tools/py.typed` — enables correct type discovery
- Align Haystack `run()` return types with `@component.output_types(...)` and returned `dict[...]` shape — Keeps Haystack component APIs type-safe and consistent with `@component.output_types(...)`, preventing misleading nullable or overly broad contracts.
- Centralize untyped import suppressions in `integrations/*/pyproject.toml` — avoids scattered `# type: ignore[import-untyped]`
- Keep each integration’s tooling in `pyproject.toml`; include every importable package in `types` — ensures all shipped integration code is type-checked
- Use direct annotations for available types — avoid unnecessary quoted strings

### Code Style

- Mark state-free helper methods `@staticmethod` — clarifies no `self`/`cls` coupling
- Use structured `logger.*` placeholders in `integrations/` — pass dynamic values as kwargs
- Delete obsolete `integrations/` artifacts — stale docs, examples, deps, and configs mislead users
- Keep `integrations/**/haystack_integrations` roots/intermediates namespace-only — omit `__init__.py` unless a concrete integration package needs exports/init
- Keep sync/async `document_store.py` methods symmetrical; share filter/count/arg/error helpers — Symmetric sync/async APIs and shared helpers reduce drift, duplicated bugs, and inconsistent behavior across integration document stores.
- Initialize external clients and `None`→instance attrs in `warm_up()` — avoids lazy runtime failures

### Dependencies

- Set explicit minimum deps in `integrations/*/pyproject.toml`; avoid pins/upper bounds unless required — Accurate lower bounds keep integrations installable on the oldest compatible stack while avoiding unnecessary resolver conflicts and premature incompatibility with newer Haystack releases.
- Use official provider Python SDKs in `integrations` when they cover the workflow — reduces provider API bugs
- Keep `integrations/*/pyproject.toml` test deps minimal — rely on inherited deps and add only imports/tests need
- Declare only directly used runtime deps in `integrations/*/pyproject.toml` — avoids bloated installs
- Use `request_with_retry`/`async_request_with_retry` for HTTP retries — avoid custom loops and expose `timeout`/`max_retries`

### Naming

- Use canonical integration names everywhere — match package names in `integrations/*`, READMEs, URLs, and tables
- Name async document-store methods `<method>_async` — keeps sync/async APIs and logs unambiguous

### General

- Import required deps at module top; reserve lazy/`try` imports for optional deps or cycles — Failing fast exposes missing required packages during import instead of hiding broken integrations until runtime.
- Re-export only intentional public API in `__init__.py` — preserves stable imports
- Wrap backend failures as `DocumentStoreError` — keep sync and async handling consistent, and preserve documented bulk write/delete error behaviour
