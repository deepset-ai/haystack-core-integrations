<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# API Docstring Style

> How to write public API and component docstrings: concise Haystack-style summaries, complete parameter/return/exception documentation, synchronized sync/async docs, performance caveats, default-value wording, inline-code markup, Markdown links, and fenced Python usage examples.

**When to check**: When writing or updating public API, component, constructor, method, or usage-example docstrings

## Rules

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
