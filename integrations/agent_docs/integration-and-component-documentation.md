<!-- Mined from deepset PR reviews; see the repo-root AGENTS.md. -->

# Integration and Component Documentation

> Documentation expectations for integrations and components, including minimal template-aligned integration READMEs, avoiding generated docs artifacts, documenting authentication modes, local test prerequisites, precedence rules, externally defined behavior, license placeholders, parameter requirements/defaults, realistic examples, and alignment with actual retrieval/scoring behavior.

**When to check**: When writing or updating READMEs, docstrings, examples, authentication docs, parameter documentation, LICENSE files, or generated documentation artifacts

## Rules

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
