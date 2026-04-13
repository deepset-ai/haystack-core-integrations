# CLAUDE.md — FalkorDB Haystack Integration: Execution Plan

> **⚠️ ALWAYS READ FIRST:** Before working in this repo, read `AGENTS.md` and follow ALL instructions there.
> All `hatch` commands must be run from inside `integrations/falkor_db/`, never from the repo root.

---

## Project Overview

**Goal:** Build `falkor-db-haystack` — an official Haystack integration for FalkorDB, a high-performance
graph database optimised for GraphRAG. The integration is modelled after the `neo4j-haystack` integration
but adapted fully to FalkorDB's OpenCypher dialect.

**Critical Constraint — No APOC:** FalkorDB does not support Neo4j's APOC library. Every query that would
use APOC (e.g. `apoc.do.when`, `apoc.create.uuids`, bulk-write helpers) must be replaced with:
- Standard OpenCypher (`UNWIND` for batching, `MERGE`/`SET` for upserts, `CASE` for conditionals)
- FalkorDB-native vector index syntax (`CALL db.idx.vector.queryNodes(...)`)

**FalkorDB Python driver:** [`falkordb`](https://pypi.org/project/falkordb/) (the official Python client).

**Package name (PyPI):** `falkor-db-haystack`  
**Namespace path:** `haystack_integrations.document_stores.falkor_db` / `haystack_integrations.components.retrievers.falkor_db`

---

## State Legend

| Symbol | Meaning |
|--------|---------|
| `- [ ]` | Not started |
| `- [~]` | In progress |
| `- [x]` | Complete |

---

## Phase 1 — Repository Infrastructure & Package Foundation

*Establish the package skeleton, dependencies, and a developer-runnable environment before any logic is written.*

### 1.1 — Finalise `pyproject.toml` dependencies and keywords

- [x] **Objective:** Add the `falkordb` Python driver as a real runtime dependency; add PyPI keywords and
  classifiers that will surface this package in GraphRAG / graph-database searches.

- **Target Files:**
  - `integrations/falkor_db/pyproject.toml`

- **Key Engineering Considerations:**
  - Pin `falkordb>=1.0,<2` (verify the latest stable semver on PyPI before pinning).
  - The `falkordb` driver wraps a Redis-like wire protocol; confirm it does **not** pull in `neo4j` as a
    transitive dep (would cause namespace confusion).
  - Add keywords: `["haystack", "falkordb", "graphrag", "graph", "vector-search", "document-store",
    "rag", "openCypher"]`.
  - Classifiers to add: `"Topic :: Database"` and `"Topic :: Scientific/Engineering :: Artificial Intelligence"`.
  - The `[tool.hatch.envs.test]` block already has `pytest`, `mypy`, etc.; add `falkordb` there too so
    tests can import it.

---

### 1.2 — Expand the source package directory tree

- [x] **Objective:** Create the full namespace package layout that mirrors the elasticsearch/chroma pattern
  so both the document store and retriever components sit under the correct namespace paths.

- **Target Files (to create):**
  ```
  integrations/falkor_db/src/haystack_integrations/
  ├── document_stores/
  │   └── falkor_db/
  │       ├── __init__.py          ← already exists (empty); needs exports added in Phase 2
  │       └── py.typed             ← already exists
  └── components/
      └── retrievers/
          └── falkor_db/
              ├── __init__.py      ← NEW
              └── py.typed         ← NEW
  ```

- **Key Engineering Considerations:**
  - The `components/` subtree does not yet exist under `integrations/falkor_db/src/`. It must mirror
    the pattern used by `integrations/elasticsearch/src/haystack_integrations/components/`.
  - Both `__init__.py` files must follow the SPDX licence header convention used across the repo:
    ```
    # SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
    #
    # SPDX-License-Identifier: Apache-2.0
    ```
  - `py.typed` files are zero-byte PEP 561 markers; they must exist in both `document_stores/falkor_db/`
    and `components/retrievers/falkor_db/`.

---

### 1.3 — Update `pydoc/config_docusaurus.yml`

- [x] **Objective:** Register all module paths (document store + both retrievers) so the pydoc renderer
  generates a complete API reference page.

- **Target Files:**
  - `integrations/falkor_db/pydoc/config_docusaurus.yml`

- **Key Engineering Considerations:**
  - Current config only lists `haystack_integrations.document_stores.falkor_db.document_store`.
  - Must add:
    - `haystack_integrations.components.retrievers.falkor_db.embedding_retriever`
    - `haystack_integrations.components.retrievers.falkor_db.cypher_retriever`
  - Keep `id: integrations-falkor_db` and `filename: falkor-db.md`; update the `description` to
    reflect all three exposed components.

---

### 1.4 — Update CI workflow for integration tests

- [x] **Objective:** Gate integration tests behind a running FalkorDB instance; add a Docker service step
  to the GHA workflow and the required secrets (if any).

- **Target Files:**
  - `.github/workflows/falkor_db.yml`

- **Key Engineering Considerations:**
  - FalkorDB is available as a Docker image: `falkordb/falkordb:latest` (listens on port `6379`).
  - No API key is needed for a self-hosted instance; integration tests connect via `host=localhost, port=6379`.
  - Add a `services:` block to the `run` job:
    ```yaml
    services:
      falkordb:
        image: falkordb/falkordb:latest
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-retries 5
    ```
  - The unit test step already runs without integration services; ensure `pytest -m "not integration"`
    remains unaffected.
  - `FALKORDB_HOST` and `FALKORDB_PORT` env vars should be injected into the test step, defaulting to
    `localhost` / `6379` so local dev also works without extra setup.

---

## Phase 2 — `FalkorDBDocumentStore`

*The core storage layer. All graph interactions funnel through this class.*

### 2.1 — Implement `FalkorDBDocumentStore`

- [x] **Objective:** A fully `DocumentStore`-protocol-compliant class that stores Haystack `Document`
  objects as graph nodes in FalkorDB, with full CRUD, filtering, and UNWIND-based batch writes.

- **Target Files (to create/modify):**
  - `integrations/falkor_db/src/haystack_integrations/document_stores/falkor_db/document_store.py` ← **CREATE**
  - `integrations/falkor_db/src/haystack_integrations/document_stores/falkor_db/__init__.py` ← **MODIFY** (add export)

- **Public API surface:**
  ```python
  class FalkorDBDocumentStore:
      def __init__(
          self,
          host: str = "localhost",
          port: int = 6379,
          graph_name: str = "haystack",
          username: str | None = None,
          password: Secret | None = None,
          embedding_dim: int = 768,
          similarity: SimilarityFunction = SimilarityFunction.COSINE,
          recreate_index: bool = False,
      ) -> None: ...

      # Haystack DocumentStore protocol
      def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int: ...
      def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]: ...
      def delete_documents(self, document_ids: list[str]) -> None: ...
      def count_documents(self) -> int: ...

      # Internal helpers (accessed by retrievers)
      def _embedding_retrieval(self, query_embedding: list[float], top_k: int, filters: ...) -> list[Document]: ...
      def _cypher_retrieval(self, cypher_query: str, parameters: dict[str, Any]) -> list[Document]: ...

      # Serialisation
      def to_dict(self) -> dict[str, Any]: ...
      @classmethod
      def from_dict(cls, data: dict[str, Any]) -> "FalkorDBDocumentStore": ...
  ```

- **Key Engineering Considerations:**

  **A) Connection management:**
  - Use `falkordb.FalkorDB(host=..., port=..., username=..., password=...)` to get the client.
  - Expose `graph_name` to allow multiple isolated graphs in a single FalkorDB instance.
  - Lazy-connect (connect on first operation) to support clean serialisation/deserialisation.

  **B) Schema / index creation — NO APOC:**
  - Neo4j-haystack uses APOC for constraints; FalkorDB uses native index syntax:
    ```cypher
    CREATE INDEX FOR (d:Document) ON (d.id)
    ```
  - Vector index creation uses FalkorDB's specific call syntax:
    ```cypher
    CALL db.idx.vector.createNodeIndex(
      'Document', 'embedding',
      $dim, 'cosine'
    )
    ```
  - Run index creation in an `_ensure_index()` method called from `__init__` (or on first write when
    `recreate_index=True`).

  **C) `write_documents` — UNWIND batch upsert (no APOC):**
  - Neo4j-haystack uses `apoc.do.when` for upsert logic. Replace with idiomatic OpenCypher `MERGE ... ON CREATE SET ... ON MATCH SET`:
    ```cypher
    UNWIND $docs AS doc
    MERGE (d:Document {id: doc.id})
    ON CREATE SET d += doc.props
    ON MATCH SET d += doc.props
    ```
  - Batch size: 100 documents per `UNWIND` to avoid memory spikes; make this a private constant
    `_WRITE_BATCH_SIZE = 100`.
  - Embedding vectors are stored as a node property `d.embedding = doc.embedding` (FalkorDB stores
    float arrays natively).
  - `DuplicatePolicy.FAIL` — use `MERGE` then check for `ON CREATE` only; raise `DuplicateDocumentError`
    if the node already existed.
  - `DuplicatePolicy.SKIP` — use `MERGE` but do not run `ON MATCH SET`.
  - `DuplicatePolicy.OVERWRITE` — use `MERGE ... ON MATCH SET d += doc.props`.

  **D) `filter_documents` — Haystack filter-to-Cypher translation:**
  - Implement a `_convert_filters(filters)` helper that translates Haystack's filter DSL
    (`field`, `operator`, `conditions`) into a `WHERE` clause string + params dict.
  - Supported operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, `AND`, `OR`, `NOT`.
  - Use parameterised queries throughout (`$param`) to prevent injection.

  **E) `delete_documents`:**
  ```cypher
  UNWIND $ids AS id
  MATCH (d:Document {id: id})
  DETACH DELETE d
  ```

  **F) Serialisation:**
  - `password` is a `haystack.utils.Secret`; serialise it the same way other integrations handle
    secrets (call `password.to_dict()` / `Secret.from_dict()`).
  - Use `@component` decorator pattern; export from `__init__.py`:
    ```python
    from .document_store import FalkorDBDocumentStore
    __all__ = ["FalkorDBDocumentStore"]
    ```

---

## Phase 3 — Retriever Components

*Two retriever components; both are thin wrappers that delegate to internal `DocumentStore` methods.*

### 3.1 — `FalkorDBEmbeddingRetriever`

- [x] **Objective:** Retrieves documents using semantic vector similarity via FalkorDB's native vector index.

- **Target Files (to create/modify):**
  - `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/embedding_retriever.py` ← **CREATE**
  - `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/__init__.py` ← **MODIFY**

- **Public API surface:**
  ```python
  @component
  class FalkorDBEmbeddingRetriever:
      def __init__(
          self,
          document_store: FalkorDBDocumentStore,
          filters: dict[str, Any] | None = None,
          top_k: int = 10,
          filter_policy: FilterPolicy = FilterPolicy.REPLACE,
      ) -> None: ...

      @component.output_types(documents=list[Document])
      def run(
          self,
          query_embedding: list[float],
          filters: dict[str, Any] | None = None,
          top_k: int | None = None,
      ) -> dict[str, list[Document]]: ...
  ```

- **Key Engineering Considerations:**

  **FalkorDB vector search syntax — NO APOC:**
  - Neo4j-haystack calls `db.index.vector.queryNodes(...)` as a Neo4j procedure. FalkorDB's equivalent
    (as of driver v1.x) is:
    ```cypher
    CALL db.idx.vector.queryNodes(
      'Document', 'embedding', $top_k, vecf32($query_embedding)
    ) YIELD node, score
    RETURN node, score
    ```
  - `vecf32()` is FalkorDB's built-in function to cast a float list to an internal vector type.
  - The `score` returned is a distance (lower = more similar for cosine); sort ascending then invert
    or normalise for Haystack's `score` field.
  - Additional `filters` must be applied as a post-query Python-side filter (or injected as a
    `WHERE node.field = $value` in the `YIELD` clause) — verify what FalkorDB supports in
    vector CALL filtering before implementing.

  **`filter_policy`:**
  - Mirror the `apply_filter_policy` pattern from `haystack.document_stores.types.filter_policy`.

---

### 3.2 — `FalkorDBCypherRetriever` (Dynamic Retriever)

- [x] **Objective:** A power-user retriever that executes an arbitrary OpenCypher query against the graph
  and maps results back to Haystack `Document` objects. Enables GraphRAG traversal patterns.

- **Target Files (to create/modify):**
  - `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/cypher_retriever.py` ← **CREATE**
  - `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/__init__.py` ← **MODIFY**

- **Public API surface:**
  ```python
  @component
  class FalkorDBCypherRetriever:
      def __init__(
          self,
          document_store: FalkorDBDocumentStore,
          custom_cypher_query: str | None = None,
      ) -> None: ...

      @component.output_types(documents=list[Document])
      def run(
          self,
          query: str | None = None,
          parameters: dict[str, Any] | None = None,
      ) -> dict[str, list[Document]]: ...
  ```

- **Key Engineering Considerations:**
  - Accept either a static `custom_cypher_query` set at init time, or a dynamic `query` passed at
    runtime — runtime `query` takes precedence if both are provided.
  - The query must return nodes or maps; the retriever calls `_cypher_retrieval()` on the store,
    which maps result rows to `Document` objects using a `_node_to_document()` helper.
  - **Security note:** Document this component as an advanced tool and note in the docstring that raw
    Cypher queries must come from trusted sources (no user-supplied strings without sanitisation).
  - `parameters` dict is passed directly as Cypher params (`$param_name` syntax) so drivers handle
    escaping — parameterised queries are always safer than string interpolation.

---

### 3.3 — Top-level `__init__.py` exports

- [x] **Objective:** Ensure all three public classes are importable from their respective top-level
  namespace packages, following the pattern of other integrations.

- **Target Files:**
  - `integrations/falkor_db/src/haystack_integrations/document_stores/falkor_db/__init__.py`
  - `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/__init__.py`

- **Key Engineering Considerations:**
  - The document store `__init__.py` should export `FalkorDBDocumentStore` and `SimilarityFunction`
    (if defined as an enum in `document_store.py`).
  - The retriever `__init__.py` should export both `FalkorDBEmbeddingRetriever` and
    `FalkorDBCypherRetriever`.
  - Verify that `hatch run test:types` (mypy) finds all exports correctly.

---

## Phase 4 — Tests

*Two test files: one for the document store, one for both retrievers. Unit tests are mocked; integration
tests require a live FalkorDB container.*

### 4.1 — Unit tests for `FalkorDBDocumentStore`

- [ ] **Objective:** Cover all public methods of the document store without a real DB, using `unittest.mock`.

- **Target Files (to create):**
  - `integrations/falkor_db/tests/test_document_store.py`

- **Key Engineering Considerations:**
  - Mock `falkordb.FalkorDB` and `falkordb.Graph` at the module level so no DB connection is needed.
  - Test cases required:
    - `test_write_documents_overwrite_policy` — verify `MERGE ... ON MATCH SET` path is taken.
    - `test_write_documents_skip_policy` — verify `ON MATCH SET` is NOT executed.
    - `test_write_documents_fail_policy` — verify `DuplicateDocumentError` raised on collision.
    - `test_write_documents_batching` — write `_WRITE_BATCH_SIZE + 1` docs; assert driver called twice.
    - `test_filter_documents_empty_filters` — no `WHERE` clause in generated Cypher.
    - `test_filter_documents_eq_operator` — `WHERE d.field = $val` generated correctly.
    - `test_filter_documents_and_operator` — nested conditions produce `AND`-joined clauses.
    - `test_delete_documents` — verify `DETACH DELETE` query is sent with correct IDs.
    - `test_count_documents` — verify `COUNT(d)` query is sent and result parsed.
    - `test_to_dict_from_dict_roundtrip` — serialisation preserves host, port, graph_name, etc.
    - `test_password_not_exposed_in_to_dict` — Secret is serialised safely (not plaintext).

---

### 4.2 — Unit tests for Retrievers

- [ ] **Objective:** Cover both retrievers' `run()` methods and serialisation without a real DB.

- **Target Files (to create):**
  - `integrations/falkor_db/tests/test_retrievers.py`

- **Key Engineering Considerations:**
  - Mock `FalkorDBDocumentStore._embedding_retrieval` and `_cypher_retrieval` to return a fixed list
    of `Document` objects.
  - Test cases required:
    - `test_embedding_retriever_run` — verify `query_embedding` is passed through correctly, result is
      `{"documents": [...]}`.
    - `test_embedding_retriever_filter_policy_replace` — runtime filters replace init filters.
    - `test_embedding_retriever_filter_policy_merge` — runtime filters merge with init filters.
    - `test_embedding_retriever_to_dict_from_dict` — round-trip preserves `top_k`, `filter_policy`.
    - `test_cypher_retriever_run_with_init_query` — static `custom_cypher_query` is used.
    - `test_cypher_retriever_run_with_runtime_query` — runtime `query` overrides static one.
    - `test_cypher_retriever_invalid_store_type` — raises `ValueError` if non-FalkorDB store passed.

---

### 4.3 — Integration tests

- [ ] **Objective:** End-to-end tests against a live FalkorDB instance (marked `@pytest.mark.integration`).

- **Target Files (to create):**
  - `integrations/falkor_db/tests/test_integration.py`

- **Key Engineering Considerations:**
  - Use a `falkordb_document_store` pytest fixture that connects to `localhost:6379` (or env vars
    `FALKORDB_HOST` / `FALKORDB_PORT`) and tears down (`DETACH DELETE` all nodes) after each test via
    a `yield` fixture.
  - Test cases required:
    - `test_write_and_filter_documents` — write 3 docs, filter by metadata field, assert correct subset returned.
    - `test_write_documents_duplicate_skip` — write same doc twice with SKIP; count returns 1.
    - `test_write_documents_duplicate_overwrite` — write, update field, overwrite; verify updated value persists.
    - `test_embedding_retrieval` — write docs with embeddings, run embedding retriever, verify top-1 is correct.
    - `test_cypher_retriever_graph_traversal` — write node with relationship to another node, run a MATCH
      traversal query, assert both nodes returned.
    - `test_delete_documents` — write then delete; count returns 0.
  - All integration tests must be decorated `@pytest.mark.integration`.
  - Add a `docker-compose.yml` at `integrations/falkor_db/docker-compose.yml` for local dev with:
    ```yaml
    services:
      falkordb:
        image: falkordb/falkordb:latest
        ports:
          - "6379:6379"
    ```

---

## Phase 5 — Polish, Documentation & Final QA

### 5.1 — Flesh out `README.md`

- [ ] **Objective:** Write a user-facing README covering installation, quickstart, and pipeline examples.

- **Target Files:**
  - `integrations/falkor_db/README.md`

- **Key Engineering Considerations:**
  - Follow the structure of `integrations/elasticsearch/README.md` as a template.
  - Sections to include:
    1. **Installation** (`pip install falkor-db-haystack`)
    2. **Prerequisites** (running FalkorDB via Docker)
    3. **Quickstart** — ingest documents with embeddings, run an embedding retrieval pipeline
    4. **GraphRAG Example** — show `FalkorDBCypherRetriever` for multi-hop graph traversal
    5. **Configuration Reference** — table of `FalkorDBDocumentStore` init parameters
  - Code examples must be runnable and use only publicly exported symbols.

---

### 5.2 — Add to root `README.md` integrations table

- [ ] **Objective:** Register FalkorDB in the main monorepo integrations table.

- **Target Files:**
  - `README.md` (repo root)

- **Key Engineering Considerations:**
  - Find the `Document Stores` section of the table and add a row:
    `| FalkorDB | falkor-db-haystack | graph database / GraphRAG | ... |`
  - Match the column format of neighbouring rows exactly (badge links, PyPI link).

---

### 5.3 — Final quality gate

- [ ] **Objective:** All checks pass before opening a PR.

- **Checklist (run from `integrations/falkor_db/`):**
  ```sh
  hatch run fmt              # auto-format & lint (ruff)
  hatch run fmt-check        # validate format
  hatch run test:types       # mypy type check
  hatch run test:unit        # unit tests (no DB)
  hatch run test:integration # integration tests (needs Docker)
  ```

- **Key Engineering Considerations:**
  - mypy target path already configured: `mypy -p haystack_integrations.document_stores.falkor_db`.
    Must also add `haystack_integrations.components.retrievers.falkor_db` to the types script in
    `pyproject.toml` `[tool.hatch.envs.test.scripts]`.
  - All public methods must have type annotations and docstrings (ruff rules D102, D103, ANN enforced).
  - `ANN401` (`Any` allowed) is already suppressed — use sparingly, only at SDK boundaries.
  - Coverage gate: 90% green / 60% orange per CI config. Ensure unit tests alone hit ≥ 90%.

---

## File Creation Summary

| File | Action |
|------|--------|
| `integrations/falkor_db/pyproject.toml` | Modify — add deps, keywords |
| `integrations/falkor_db/src/haystack_integrations/document_stores/falkor_db/__init__.py` | Modify — add exports |
| `integrations/falkor_db/src/haystack_integrations/document_stores/falkor_db/document_store.py` | **Create** |
| `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/__init__.py` | **Create** |
| `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/py.typed` | **Create** |
| `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/embedding_retriever.py` | **Create** |
| `integrations/falkor_db/src/haystack_integrations/components/retrievers/falkor_db/cypher_retriever.py` | **Create** |
| `integrations/falkor_db/tests/test_document_store.py` | **Create** |
| `integrations/falkor_db/tests/test_retrievers.py` | **Create** |
| `integrations/falkor_db/tests/test_integration.py` | **Create** |
| `integrations/falkor_db/docker-compose.yml` | **Create** |
| `integrations/falkor_db/pydoc/config_docusaurus.yml` | Modify — add retriever modules |
| `integrations/falkor_db/README.md` | Modify — full content |
| `.github/workflows/falkor_db.yml` | Modify — add Docker service for integration tests |
| `README.md` (repo root) | Modify — add FalkorDB row to integrations table |

---

*Last updated: 2026-04-13 — Phase 1 ✅ Phase 2 ✅. Ready for Phase 3.*
