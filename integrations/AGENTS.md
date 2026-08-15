<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# integrations/ Guidelines

## Testing

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

## API Design

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

## Config

- Align `integrations/*/pyproject.toml` Hatch docs config — use `haystack-pydoc pydoc/config_docusaurus.yml` and only docs/lint deps like `haystack-pydoc-tools` and `ruff`
- Populate `project.keywords` in each `integrations/*/pyproject.toml` — improves package discoverability
- Align `integrations/*/pyproject.toml` Python metadata with tested support — prevents invalid installs and stale compatibility claims
- Align `integrations/<component>/pyproject.toml` Ruff config with the shared template; put test-only ignores under `[tool.ruff.lint.per-file-ignores]` for `"tests/**/*"` — keeps integration linting consistent without weakening global rules
- Align `integrations/*/pyproject.toml` with the canonical template — ensures consistent packaging
- Set real `[project].authors` in `integrations/*/pyproject.toml` — captures true maintainer and partner ownership
- Set `description` in `integrations/*/pyproject.toml` to approved integration wording — avoids vague package metadata

## Type System

- Prefer type-correct code over `# type: ignore`; if needed, use exact-line `# type: ignore[code]` with a safety comment — Type-correct code avoids hiding real bugs, while targeted suppressions keep unavoidable checker limitations auditable and safe.
- Place integration `py.typed` at the exposed package boundary, e.g. `haystack_integrations/tools/py.typed` — enables correct type discovery
- Align Haystack `run()` return types with `@component.output_types(...)` and returned `dict[...]` shape — Keeps Haystack component APIs type-safe and consistent with `@component.output_types(...)`, preventing misleading nullable or overly broad contracts.
- Centralize untyped import suppressions in `integrations/*/pyproject.toml` — avoids scattered `# type: ignore[import-untyped]`
- Keep each integration’s tooling in `pyproject.toml`; include every importable package in `types` — ensures all shipped integration code is type-checked
- Use direct annotations for available types — avoid unnecessary quoted strings

## Code Style

- Mark state-free helper methods `@staticmethod` — clarifies no `self`/`cls` coupling
- Use structured `logger.*` placeholders in `integrations/` — pass dynamic values as kwargs
- Delete obsolete `integrations/` artifacts — stale docs, examples, deps, and configs mislead users
- Keep `integrations/**/haystack_integrations` roots/intermediates namespace-only — omit `__init__.py` unless a concrete integration package needs exports/init
- Keep sync/async `document_store.py` methods symmetrical; share filter/count/arg/error helpers — Symmetric sync/async APIs and shared helpers reduce drift, duplicated bugs, and inconsistent behavior across integration document stores.
- Initialize external clients and `None`→instance attrs in `warm_up()` — avoids lazy runtime failures

## Dependencies

- Set explicit minimum deps in `integrations/*/pyproject.toml`; avoid pins/upper bounds unless required — Accurate lower bounds keep integrations installable on the oldest compatible stack while avoiding unnecessary resolver conflicts and premature incompatibility with newer Haystack releases.
- Use official provider Python SDKs in `integrations` when they cover the workflow — reduces provider API bugs
- Keep `integrations/*/pyproject.toml` test deps minimal — rely on inherited deps and add only imports/tests need
- Declare only directly used runtime deps in `integrations/*/pyproject.toml` — avoids bloated installs
- Use `request_with_retry`/`async_request_with_retry` for HTTP retries — avoid custom loops and expose `timeout`/`max_retries`

## Naming

- Use canonical integration names everywhere — match package names in `integrations/*`, READMEs, URLs, and tables
- Name async document-store methods `<method>_async` — keeps sync/async APIs and logs unambiguous

## General

- Import required deps at module top; reserve lazy/`try` imports for optional deps or cycles — Failing fast exposes missing required packages during import instead of hiding broken integrations until runtime.
- Re-export only intentional public API in `__init__.py` — preserves stable imports
- Wrap backend failures as `DocumentStoreError` — keep sync and async handling consistent, and preserve documented bulk write/delete error behaviour

## Topic Guides

Check these when working in specific areas:

- **[Integration and Component Documentation](agent_docs/integration-and-component-documentation.md)**: When writing or updating READMEs, docstrings, examples, authentication docs, parameter documentation, LICENSE files, or generated documentation artifacts
