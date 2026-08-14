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
guidance referring to APIs removed or moved in Haystack 3.0 does not survive. Each
`<!-- rule:N -->` marker traces back to the review comments it came from.

They describe what reviewers actually enforce. Follow them the way you would follow a
reviewer's note: they encode reasons, not ceremony, so when a rule genuinely does not
fit the change at hand, say why rather than contorting the code to satisfy it.

## API Design

<!-- rule:-1 -->
- Target Haystack 3.x APIs: use `ChatGenerator`s, create clients in `warm_up()`, and support the module allowlist — Keeps integrations compatible with Haystack 3.x lifecycle, chat APIs, and safe deserialization behavior.
<!-- rule:60 -->
- Keep `to_dict()`/`from_dict()` symmetric with `__init__` — preserve all runtime config, including Watsonx `max_retries`, `meta_fields_to_embed`, and `embedding_separator`.
<!-- rule:37 -->
- Keep Haystack `__init__` light; route reusable setup through `warm_up()` — prevents slow construction and duplicated lifecycle logic, especially in `integrations/mcp/src/haystack_integrations/tools/mcp/`
<!-- rule:377 -->
- Preserve protocol parameter order; append store-specific args and pass ambiguous optionals by keyword — Maintains protocol compatibility while allowing implementations to add store-specific options without breaking callers.
<!-- rule:177 -->
- Align public API names across signatures, docs, code, and returns — document intentional mismatches
<!-- rule:66 -->
- Make required `run(...)` inputs explicit keyword-only args — prevents no-op component runs
<!-- rule:106 -->
- Use `StreamingChunk.reasoning`/`ReasoningContent` for reasoning — keep `meta` incidental
<!-- rule:10 -->
- Resolve `Secret` values in `warm_up`, not `__init__` — avoids init-time side effects
<!-- rule:3 -->
- Use Haystack serialization shape with runtime `type` — e.g. `{"type": generate_qualified_class_name(type(self)), "data": ...}`
<!-- rule:50 -->
- Align `Document` embedder constructors in `haystack/components/embedders` and `integrations/*/src/haystack_integrations/components/embedders/*` — include provider-relevant options like `meta_fields_to_embed`, `embedding_separator`, `prefix`, `suffix`, and `batch_size` when comparable embedders expose them
<!-- rule:231 -->
- Make `warm_up()` idempotent with an `__init__` flag set only after setup succeeds — This prevents repeated expensive setup and avoids partially marking failed initialization as ready.
<!-- rule:521 -->
- Type chat generator `tools` as `ToolsType`; don’t narrow provider-native params — preserves Haystack API compatibility and tool pass-through

## Documentation

<!-- rule:77 -->
- Keep comments concise and substantive — explain non-obvious intent, limits, edge cases, workarounds, or real compat needs
<!-- rule:221 -->
- Update `pydoc/config_docusaurus.yml` for public API changes — include modules, retrievers, and errors
<!-- rule:384 -->
- Keep docstrings authoritative and preserve reference links — prevents stale or duplicated docs
<!-- rule:46 -->
- Add the standard `SPDX-FileCopyrightText` and `Apache-2.0` header before imports — Ensures repository-wide license compliance and keeps source and test files consistent across core and integration packages.
<!-- rule:32 -->
- Keep public examples current with supported APIs — update docstrings, cookbooks, integrations, Google GenAI model names, and `RagasEvaluator` `ragas.metrics.collections` usage

## Config

<!-- rule:337 -->
- Source workflow creds from matching `${{ secrets.<SECRET_NAME> }}` env vars — keeps CI secure and reliable
<!-- rule:40 -->
- Align workflow `python-version` matrices to min/max supported Python — catches boundary regressions
<!-- rule:55 -->
- Default secrets from provider env vars; use `WATSONX_API_KEY` for Watsonx components — Predictable env-var defaults make credentials easy to configure across integrations while avoiding hardcoded secrets or setup surprises.
<!-- rule:497 -->
- Use SDK env var names consistently across CI, tests, secrets, and skips — prevents config drift
<!-- rule:412 -->
- Store credential fields as Haystack `Secret`, not `str` — avoids leaking sensitive config

## Testing

<!-- rule:123 -->
- Gate `integrations/<provider>/tests/` explicitly — wire CI coverage, inject `Secret`/env values, use provider-specific `pytest.mark.skipif(...)` reasons, precise `sys.version_info`/`sys.platform` checks, and validate external-API tests locally with a personal key.
<!-- rule:127 -->
- Test each advertised format variant, including optional params like `embedding_types` — catches non-default response and content bugs
<!-- rule:41 -->
- Include all supported OSes in `.github/workflows/` test matrices — catches OS-specific bugs

## Code Style

<!-- rule:264 -->
- Update Haystack `Document`s with `dataclasses.replace(...)` — avoids in-place mutation bugs
<!-- rule:364 -->
- Exclude generated artifacts from feature PRs — let release/merge workflows create them
<!-- rule:223 -->
- Inline tiny private helpers used once — keeps control flow clear and avoids indirection

## Dependencies

<!-- rule:-3 -->
- Run `hatch` inside `integrations/<name>/` — each integration owns its envs
<!-- rule:380 -->
- Pin `.github/workflows/` Action `uses:` deps to full commit SHAs — avoid mutable refs

## Imports

<!-- rule:-2 -->
- Import moved 3.0 integrations from `haystack_integrations`, not `haystack` — avoids broken imports
<!-- rule:43 -->
- Import `from haystack import logging` for project logging; keep `logging.getLogger(__name__)` — This uses Haystack’s logging behavior consistently across tests, components, and integrations while preserving standard logger naming.

## General

<!-- rule:322 -->
- Type inputs/returns precisely; replace `Any` when supported values or shapes are known — Precise structures and element types improve type checking, document API contracts, and prevent shape-related bugs.
<!-- rule:312 -->
- Use `metadata_field`/`metadata_fields` in public APIs — clarifies document metadata args
<!-- rule:366 -->
- Raise `TypeError` for wrong input types/shapes; `ValueError` for invalid values/config — This preserves consistent API misuse semantics and helps callers distinguish type errors from invalid values.

## File-Specific Rules

### `README.md`
<!-- rule:346 -->
- Sort the `README.md` integrations table alphabetically — keeps entries findable and diffs clean

## API Docstring Style

_When to check: When writing or updating public API, component, constructor, method, or usage-example docstrings_

<!-- rule:234 -->
- Use single backticks for inline code in prose; reserve double backticks for Haystack release notes — Single-backtick inline code renders consistently across docstrings and docs, while preserving the Haystack release-note exception.
<!-- rule:19 -->
- Document all public callable params; put explicit `__init__` params in the `__init__` docstring — Keeps public API docs accurate and ensures generated documentation shows constructor args in the expected place.
<!-- rule:532 -->
- Document performance caveats in API docstrings — warn about slow queries or large scans
<!-- rule:198 -->
- Write concise public docstrings and sync async variants — document real contracts, not internals
<!-- rule:405 -->
- Use single-line Markdown links in docstrings/comments — keeps docs readable and links rendering
<!-- rule:274 -->
- Document public returns with Sphinx `:returns:` — include mapping keys/types and match `@component.output_types(...)` names
<!-- rule:393 -->
- Describe `__init__` params by purpose/constraints; omit defaults already in signatures — Function signatures already expose defaults, so repeating them in docstrings creates stale documentation when defaults change.
<!-- rule:29 -->
- Document public method exceptions with concrete `:raises:` conditions — no `If ...` placeholders
<!-- rule:199 -->
- Write Haystack-style docstrings — one-line summary, blank line, then unindented sections/examples
<!-- rule:329 -->
- Use `### Usage example` and fenced ` ```python ` blocks for Python examples — avoids renderer issues

## Integration and Component Documentation

_When to check: When writing or updating READMEs, docstrings, examples, authentication docs, parameter documentation, LICENSE files, or generated documentation artifacts_

<!-- rule:6 -->
- Keep `integrations/*/README.md` minimal and template-aligned — link to canonical docs/examples instead of duplicating long content
<!-- rule:301 -->
- Leave generated docs under `integrations/` untouched — pipelines own `CHANGELOG.md` and API docs
<!-- rule:193 -->
- Document auth by credential presence and `Secret` inputs — prevents unclear setup paths
<!-- rule:163 -->
- Document only needed local test prerequisites in `integrations/{integration}/README.md` — avoids misleading setup
<!-- rule:63 -->
- Document option precedence in `integrations` docs — list named options overridden by maps/headers
<!-- rule:5 -->
- Document model-dependent APIs with canonical links — clarify supported kwargs, models, defaults, modes, outputs, and whether lists are exhaustive
<!-- rule:247 -->
- Fill `integrations/*/LICENSE.txt` copyright fields — use real years and holders, not placeholders
<!-- rule:205 -->
- Document non-obvious public API requirements — include ranges, external limits, syntax, and examples
<!-- rule:62 -->
- Sync constructor docs, defaults, and `None` fallbacks — state env/base/upstream defaults
<!-- rule:143 -->
- Document integrations with context-complete examples — show required `Document` setup, workflow execution, and output production
<!-- rule:185 -->
- Align integration retriever docstrings with actual retrieval/scoring support — avoids misleading users

## Directory-specific conventions

### `integrations/`

<!-- rule:360 -->
- Put async document-store tests in `test_document_store_async.py` — keeps sync/async coverage clear
<!-- rule:215 -->
- Parametrize duplicate `pytest` tests — merge same behavior into one test with `@pytest.mark.parametrize`
<!-- rule:265 -->
- Base document store tests on `haystack.testing.document_store` classes — avoids duplicate, incomplete contract tests
<!-- rule:188 -->
- Test `to_dict()`/`from_dict()` round trips with non-default init params — preserve serializable config
<!-- rule:144 -->
- Test both sync and async integration paths — mirror gates, fixtures, inputs, and assertions
<!-- rule:246 -->
- Store fixtures in `integrations/{integration}/tests/test_files/` — keeps tests local, stable, and isolated
<!-- rule:82 -->
- Consolidate `integrations/<provider>/tests/` by component — add document-store coverage to `test_document_store.py`, not one-off files
<!-- rule:114 -->
- Test only changed integration chat behavior in `integrations/*/tests/test_*chat_generator*.py` — avoids redundant live/core coverage
<!-- rule:313 -->
- Avoid routine `warm_up()` in `integrations/*/tests` init tests — call it only when asserting warm-up behavior
<!-- rule:83 -->
- Assert persisted outcomes after mutating document-store tests — catches cleanup/index regressions
<!-- rule:91 -->
- Delete unused test fixtures/helpers/setup — keeps integration tests focused and maintainable
<!-- rule:172 -->
- Test chat helper conversions directly — cover provider-specific reasoning/thinking content
<!-- rule:56 -->
- Align pytest markers in `integrations/*/pyproject.toml` — declare only used markers and set `--strict-markers`
<!-- rule:401 -->
- Test converter skip/failure paths and warnings in `integrations/*/tests/` — preserves graceful-failure behavior
<!-- rule:103 -->
- Test same-turn multi-tool calls in chat generator integrations — model them in one assistant message
<!-- rule:165 -->
- Keep test comments specific and current — preserves intent and prevents stale guidance
<!-- rule:230 -->
- Test mixed chat tools across init/runtime paths — assert merged `config.tools` and mirror sync/async coverage
<!-- rule:2 -->
- Pass explicit init args in `integrations` tests — include `model`/backend IDs to validate custom paths
<!-- rule:358 -->
- Test `close()`/reopen changes in `integrations/*/tests/test_document_store.py` — keeps lifecycle coverage consistent
<!-- rule:243 -->
- Use local `pytest` fixtures only for shared non-trivial setup — keeps tests clear and uncoupled
<!-- rule:79 -->
- Test provider streaming end-to-end in `integrations/*/tests/test_chat_generator.py` — assert every `StreamingChunk`, metadata/usage/finish field, tool-call/reasoning output, and final `ChatMessage` from realistic provider chunk sequences.
<!-- rule:7 -->
- Update `integrations/amazon_bedrock/tests/` with generator changes — cover full requests, feature paths, and config deserialization
<!-- rule:507 -->
- Test only real legacy serialization formats — avoid fake shims for missing current fields
<!-- rule:25 -->
- Assert secret-backed credentials restore and resolve — prevents hidden credential bugs in tests
<!-- rule:0 -->
- Use keyword-only args for optional public API params — preserves backward compatibility
<!-- rule:530 -->
- Use case-insensitive literal substrings for metadata/search filters; in IBM DB use `LOCATE(UPPER(?), UPPER(column)) > 0`, not `LIKE` — Case-normalized literal matching keeps search behavior consistent across integrations and prevents `%`/`_` wildcard bugs in IBM DB.
<!-- rule:48 -->
- Use `Secret` for sensitive integration config/API values — prevents leaked credentials
<!-- rule:157 -->
- Prefix non-public helpers in `integrations` with `_` — clarifies API boundaries
<!-- rule:85 -->
- Add async APIs only for native async I/O; mirror sync contracts and tests — Keeps async APIs non-blocking and consistent with sync behavior, preventing event-loop stalls and contract drift.
<!-- rule:255 -->
- Keep public Document Store APIs consistent across backends — preserves portability
<!-- rule:71 -->
- Expose only wired, supported public params; reject or document unsupported filters/flags — Avoids misleading no-op APIs, runtime surprises, and inconsistent integration behavior.
<!-- rule:217 -->
- Use Haystack default serialization when `init_parameters` can rebuild the component — avoids brittle custom `to_dict`/`from_dict`; require valid `init_parameters` for custom deserialization.
<!-- rule:94 -->
- Set `ToolCallDelta.index` from provider-stable call IDs — preserves chunk correlation
<!-- rule:419 -->
- Accept `meta` as `dict | list[dict] | None` for multi-`sources` converters — keeps integration metadata semantics consistent
<!-- rule:421 -->
- Merge `ByteStream.meta` into converter `Document.meta` and document it — preserves metadata
<!-- rule:417 -->
- Declare `SUPPORTED_MODELS` beside limited integration components — documents model limits
<!-- rule:434 -->
- Use backend-native bulk APIs in integration document stores — improves throughput and avoids race-prone per-document logic
<!-- rule:86 -->
- Preserve provider-native stream indices in `StreamingChunk` — don’t hardcode or reshape for helpers
<!-- rule:293 -->
- Apply filters before iterating, aggregating, or paginating docs — reuse `filter_documents(filters=filters)` to prevent query bugs
<!-- rule:53 -->
- Use `streaming_callback` for `StreamingChunk`s — don't return chunks in outputs; preserve metadata, handle unsupported chunk shapes explicitly, and test streaming/non-streaming paths
<!-- rule:418 -->
- Validate concrete backend deps in `__init__` — fail fast when objects are incompatible
<!-- rule:395 -->
- Preserve converter source provenance — use original paths or `ByteStream.meta['file_path']`, not synthetic temp filenames
<!-- rule:1 -->
- Expose reusable integrations as named importable APIs — e.g., `@component` `GitHubFileEditor` plus `GitHubFileEditorTool`
<!-- rule:416 -->
- Use `filter_policy` and `apply_filter_policy(...)` for retriever filters — avoids inconsistent merge bugs
<!-- rule:242 -->
- Align `integrations/*/pyproject.toml` Hatch docs config — use `haystack-pydoc pydoc/config_docusaurus.yml` and only docs/lint deps like `haystack-pydoc-tools` and `ruff`
<!-- rule:455 -->
- Populate `project.keywords` in each `integrations/*/pyproject.toml` — improves package discoverability
<!-- rule:52 -->
- Align `integrations/*/pyproject.toml` Python metadata with tested support — prevents invalid installs and stale compatibility claims
<!-- rule:51 -->
- Align `integrations/<component>/pyproject.toml` Ruff config with the shared template; put test-only ignores under `[tool.ruff.lint.per-file-ignores]` for `"tests/**/*"` — keeps integration linting consistent without weakening global rules
<!-- rule:38 -->
- Align `integrations/*/pyproject.toml` with the canonical template — ensures consistent packaging
<!-- rule:49 -->
- Set real `[project].authors` in `integrations/*/pyproject.toml` — captures true maintainer and partner ownership
<!-- rule:391 -->
- Set `description` in `integrations/*/pyproject.toml` to approved integration wording — avoids vague package metadata
<!-- rule:35 -->
- Prefer type-correct code over `# type: ignore`; if needed, use exact-line `# type: ignore[code]` with a safety comment — Type-correct code avoids hiding real bugs, while targeted suppressions keep unavoidable checker limitations auditable and safe.
<!-- rule:162 -->
- Place integration `py.typed` at the exposed package boundary, e.g. `haystack_integrations/tools/py.typed` — enables correct type discovery
<!-- rule:306 -->
- Align Haystack `run()` return types with `@component.output_types(...)` and returned `dict[...]` shape — Keeps Haystack component APIs type-safe and consistent with `@component.output_types(...)`, preventing misleading nullable or overly broad contracts.
<!-- rule:58 -->
- Centralize untyped import suppressions in `integrations/*/pyproject.toml` — avoids scattered `# type: ignore[import-untyped]`
<!-- rule:54 -->
- Keep each integration’s tooling in `pyproject.toml`; include every importable package in `types` — ensures all shipped integration code is type-checked
<!-- rule:471 -->
- Use direct annotations for available types — avoid unnecessary quoted strings
<!-- rule:76 -->
- Mark state-free helper methods `@staticmethod` — clarifies no `self`/`cls` coupling
<!-- rule:47 -->
- Use structured `logger.*` placeholders in `integrations/` — pass dynamic values as kwargs
<!-- rule:332 -->
- Delete obsolete `integrations/` artifacts — stale docs, examples, deps, and configs mislead users
<!-- rule:36 -->
- Keep `integrations/**/haystack_integrations` roots/intermediates namespace-only — omit `__init__.py` unless a concrete integration package needs exports/init
<!-- rule:285 -->
- Keep sync/async `document_store.py` methods symmetrical; share filter/count/arg/error helpers — Symmetric sync/async APIs and shared helpers reduce drift, duplicated bugs, and inconsistent behavior across integration document stores.
<!-- rule:343 -->
- Initialize external clients and `None`→instance attrs in `warm_up()` — avoids lazy runtime failures
<!-- rule:17 -->
- Set explicit minimum deps in `integrations/*/pyproject.toml`; avoid pins/upper bounds unless required — Accurate lower bounds keep integrations installable on the oldest compatible stack while avoiding unnecessary resolver conflicts and premature incompatibility with newer Haystack releases.
<!-- rule:492 -->
- Use official provider Python SDKs in `integrations` when they cover the workflow — reduces provider API bugs
<!-- rule:210 -->
- Keep `integrations/*/pyproject.toml` test deps minimal — rely on inherited deps and add only imports/tests need
<!-- rule:18 -->
- Declare only directly used runtime deps in `integrations/*/pyproject.toml` — avoids bloated installs
<!-- rule:463 -->
- Use `request_with_retry`/`async_request_with_retry` for HTTP retries — avoid custom loops and expose `timeout`/`max_retries`
<!-- rule:181 -->
- Use canonical integration names everywhere — match package names in `integrations/*`, READMEs, URLs, and tables
<!-- rule:250 -->
- Name async document-store methods `<method>_async` — keeps sync/async APIs and logs unambiguous
<!-- rule:195 -->
- Import required deps at module top; reserve lazy/`try` imports for optional deps or cycles — Failing fast exposes missing required packages during import instead of hiding broken integrations until runtime.
<!-- rule:238 -->
- Re-export only intentional public API in `__init__.py` — preserves stable imports
<!-- rule:227 -->
- Wrap document-store backend failures as `DocumentStoreError` — keep sync/async handling consistent and preserve Elasticsearch bulk write/delete `try`/`except` behavior unless intentionally documented

<!-- braindump:end -->
