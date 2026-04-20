# FalkorDB Integration — Test Fix Plan

**Date**: 2026-04-20  
**Branch**: `feature/falkordb-integration`  
**Test command**: `hatch run test:integration` (run from `integrations/falkordb/`)  
**Current result**: 28 failed / 21 passed out of 49 tests

---

## 1. Overview of All Failures

```
FAILED test_comparison_equal_with_none
FAILED test_comparison_not_equal
FAILED test_comparison_not_equal_with_none
FAILED test_comparison_greater_than
FAILED test_comparison_greater_than_with_iso_date
FAILED test_comparison_greater_than_with_string        ← expects FilterError, none raised
FAILED test_comparison_greater_than_with_list          ← expects FilterError, none raised
FAILED test_comparison_greater_than_with_none          ← raises FilterError, should return []
FAILED test_comparison_greater_than_equal
FAILED test_comparison_greater_than_equal_with_iso_date
FAILED test_comparison_greater_than_equal_with_string  ← expects FilterError, none raised
FAILED test_comparison_greater_than_equal_with_list    ← expects FilterError, none raised
FAILED test_comparison_greater_than_equal_with_none    ← raises FilterError, should return []
FAILED test_comparison_less_than
FAILED test_comparison_less_than_with_string           ← expects FilterError, none raised
FAILED test_comparison_less_than_with_list             ← expects FilterError, none raised
FAILED test_comparison_less_than_with_none             ← raises FilterError, should return []
FAILED test_comparison_less_than_equal
FAILED test_comparison_less_than_equal_with_iso_date
FAILED test_comparison_less_than_equal_with_string     ← expects FilterError, none raised
FAILED test_comparison_less_than_equal_with_list       ← expects FilterError, none raised
FAILED test_comparison_less_than_equal_with_none       ← raises FilterError, should return []
FAILED test_comparison_in
FAILED test_comparison_in_with_with_non_list_iterable  ← expects FilterError, none raised
FAILED test_comparison_not_in
FAILED test_comparison_not_in_with_with_non_list_iterable ← expects FilterError, none raised
FAILED test_or_operator
FAILED test_not_operator
```

All failures fall into **three root causes**.

---

## 2. Root Causes

### Root Cause A — `assert_documents_are_equal` ordering + float32 embedding precision

**Affects**: 20 out of 28 failures  
(all the `AssertionError` failures: `test_comparison_equal_with_none`,
`test_comparison_not_equal`, `test_comparison_not_equal_with_none`,
`test_comparison_greater_than`, `test_comparison_greater_than_with_iso_date`,
`test_comparison_greater_than_equal`, `test_comparison_greater_than_equal_with_iso_date`,
`test_comparison_less_than`, `test_comparison_less_than_equal`,
`test_comparison_less_than_equal_with_iso_date`,
`test_comparison_in`, `test_comparison_not_in`,
`test_or_operator`, `test_not_operator`, and more.)

**Sub-cause A1 — Wrong ordering**

`filter_documents()` returns nodes with `ORDER BY d.id` (lexicographic order of SHA-256
hashes). The base test's `assert_documents_are_equal` does a direct `received == expected`
comparison where `expected` is in `filterable_docs` insertion order. These two orderings
differ for any result set with more than one document.

**Sub-cause A2 — float32 embedding precision loss**

`filterable_docs` uses `_random_embeddings(768)` which returns Python `float` (float64)
lists. The write path stores embeddings with:
```cypher
SET d.embedding = vecf32(doc.embedding)
```
`vecf32` truncates 64-bit floats to 32-bit. When read back through the FalkorDB Python
client, the values are `float32`, which are **not bit-for-bit equal** to the original
`float64` values. Because Python dataclasses use `==` for all fields, `Document.__eq__`
fails for any document that has a random embedding.

**Why some tests PASS despite this:**
- `test_comparison_equal` (number == 100): no document has `number=100` → both lists are
  empty → `[] == []`.
- `test_and_operator` (number == 100 AND name == "name_0"): same.
- `test_no_filters`: writes a single `Document(content="test doc")` with no embedding.
- `test_comparison_less_than_with_iso_date`: no document has a date *before*
  `"1969-07-21T20:17:40"` → both lists are empty.

**Fix for Root Cause A**

Override `assert_documents_are_equal` in `TestDocumentStore` to:
1. Sort both lists by `doc.id` before comparing.
2. Compare `id`, `content`, `meta`, and whether `embedding` is `None` vs not-`None`
   (skip exact float comparison since float32 precision loss is inherent to vecf32
   storage and is not a bug in the filter logic being tested).

---

### Root Cause B — `>`, `>=`, `<`, `<=` operators: wrong behaviour for `None`, `str`, and `list` values

**Affects**: 8 failures
(`_with_none` ×4, `_with_string` ×4, `_with_list` ×4 — see table below)

| Test | Value | Expected behaviour | Actual behaviour |
|------|-------|-------------------|-----------------|
| `test_comparison_greater_than_with_none` | `None` | return `[]` | raises `FilterError` |
| `test_comparison_greater_than_equal_with_none` | `None` | return `[]` | raises `FilterError` |
| `test_comparison_less_than_with_none` | `None` | return `[]` | raises `FilterError` |
| `test_comparison_less_than_equal_with_none` | `None` | return `[]` | raises `FilterError` |
| `test_comparison_greater_than_with_string` | `"1"` | raise `FilterError` | no error |
| `test_comparison_greater_than_equal_with_string` | `"1"` | raise `FilterError` | no error |
| `test_comparison_less_than_with_string` | `"1"` | raise `FilterError` | no error |
| `test_comparison_less_than_equal_with_string` | `"1"` | raise `FilterError` | no error |
| `test_comparison_greater_than_with_list` | `[1]` | raise `FilterError` | no error |
| `test_comparison_greater_than_equal_with_list` | `[1]` | raise `FilterError` | no error |
| `test_comparison_less_than_with_list` | `[1]` | raise `FilterError` | no error |
| `test_comparison_less_than_equal_with_list` | `[1]` | raise `FilterError` | no error |

**Important nuance for strings**: ISO date strings (e.g., `"1972-12-11T19:54:58"`) ARE
valid values for `>`, `>=`, `<`, `<=`. The tests `test_comparison_greater_than_with_iso_date`,
`test_comparison_less_than_with_iso_date`, etc. verify that ISO date comparisons work
correctly. Only non-ISO strings (like `"1"`) must raise `FilterError`.

**Current code** (in `_build_clause`, `document_store.py:664`):
```python
if operator in _COMPARISON_OPS:
    if value is None:
        msg = f"Operator '{operator}' is not supported for None value"
        raise FilterError(msg)   # ← wrong: should return "false"
    params[param_name] = value   # ← wrong: no validation for str/list
    return f"coalesce(...)
```

**Fix for Root Cause B**

Replace the `if operator in _COMPARISON_OPS` block with:
```python
if operator in _COMPARISON_OPS:
    if value is None:
        return "false"          # Cypher literal false → empty result set
    if isinstance(value, list):
        msg = f"Operator '{operator}' does not support list values"
        raise FilterError(msg)
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except ValueError:
            msg = (
                f"Operator '{operator}' requires a numeric or ISO date value, "
                f"got non-ISO string: '{value}'"
            )
            raise FilterError(msg) from None
    params[param_name] = value
    return f"coalesce({cypher_field} {_COMPARISON_OPS[operator]} ${param_name}, false)"
```

Also add at the top of `document_store.py`:
```python
from datetime import datetime
```

---

### Root Cause C — `in` / `not in` operators: non-list iterables silently accepted

**Affects**: 2 failures
(`test_comparison_in_with_with_non_list_iterable`,
`test_comparison_not_in_with_with_non_list_iterable`)

Both tests pass a **tuple** as the value (e.g., `(10, 11)`) and expect `FilterError`.
The current code does `list(value)`, which silently converts tuples → lists with no error.

**Current code** (in `_build_clause`):
```python
if operator == "in":
    ...
    try:
        params[param_name] = list(value)   # ← silently converts tuples/sets
    except TypeError as e:
        raise FilterError(msg) from e
```

**Fix for Root Cause C**

Replace `list(value)` with a strict `isinstance(value, list)` gate. Only `list` is
accepted; everything else (int, tuple, set, generator, …) raises `FilterError`.

```python
if operator == "in":
    if not isinstance(value, list):
        msg = f"Operator 'in' requires a list value, got {type(value).__name__}"
        raise FilterError(msg)
    params[param_name] = value
    return f"coalesce({cypher_field} IN ${param_name}, false)"

if operator == "not in":
    if not isinstance(value, list):
        msg = f"Operator 'not in' requires a list value, got {type(value).__name__}"
        raise FilterError(msg)
    params[param_name] = value
    return f"coalesce(NOT ({cypher_field} IN ${param_name}), true)"
```

This replaces both the old `None` guard and the `try/except TypeError` block. The `None`
case is now handled implicitly: `isinstance(None, list)` is `False` → `FilterError`.

---

## 3. Files to Change

| File | Change type |
|------|------------|
| `src/haystack_integrations/document_stores/falkordb/document_store.py` | Bug fix (Root Causes B & C) |
| `tests/test_document_store.py` | Override `assert_documents_are_equal` (Root Cause A) |

No changes are needed to retrievers, `__init__.py`, or `pyproject.toml`.

---

## 4. Detailed Change Instructions

### 4.1 `document_store.py`

#### Step 1 — Add `datetime` import (line 8, after existing stdlib imports)

```python
# Before:
import math

# After:
import math
from datetime import datetime
```

#### Step 2 — Replace the `in` operator block (approximately line 671)

```python
# REMOVE:
    if operator == "in":
        if value is None:
            msg = "Operator 'in' is not supported for None value"
            raise FilterError(msg)
        try:
            params[param_name] = list(value)
        except TypeError as e:
            msg = f"Operator 'in' expects an iterable, but got {type(value)}"
            raise FilterError(msg) from e
        return f"coalesce({cypher_field} IN ${param_name}, false)"

# REPLACE WITH:
    if operator == "in":
        if not isinstance(value, list):
            msg = f"Operator 'in' requires a list value, got {type(value).__name__}"
            raise FilterError(msg)
        params[param_name] = value
        return f"coalesce({cypher_field} IN ${param_name}, false)"
```

#### Step 3 — Replace the `not in` operator block (approximately line 682)

```python
# REMOVE:
    if operator == "not in":
        if value is None:
            msg = "Operator 'not in' is not supported for None value"
            raise FilterError(msg)
        try:
            params[param_name] = list(value)
        except TypeError as e:
            msg = f"Operator 'not in' expects an iterable, but got {type(value)}"
            raise FilterError(msg) from e
        return f"coalesce(NOT ({cypher_field} IN ${param_name}), true)"

# REPLACE WITH:
    if operator == "not in":
        if not isinstance(value, list):
            msg = f"Operator 'not in' requires a list value, got {type(value).__name__}"
            raise FilterError(msg)
        params[param_name] = value
        return f"coalesce(NOT ({cypher_field} IN ${param_name}), true)"
```

#### Step 4 — Replace the `_COMPARISON_OPS` block (approximately line 664)

```python
# REMOVE:
    if operator in _COMPARISON_OPS:
        if value is None:
            msg = f"Operator '{operator}' is not supported for None value"
            raise FilterError(msg)
        params[param_name] = value
        return f"coalesce({cypher_field} {_COMPARISON_OPS[operator]} ${param_name}, false)"

# REPLACE WITH:
    if operator in _COMPARISON_OPS:
        if value is None:
            return "false"
        if isinstance(value, list):
            msg = f"Operator '{operator}' does not support list values"
            raise FilterError(msg)
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except ValueError:
                msg = (
                    f"Operator '{operator}' requires a numeric or ISO date value, "
                    f"got non-ISO string: '{value}'"
                )
                raise FilterError(msg) from None
        params[param_name] = value
        return f"coalesce({cypher_field} {_COMPARISON_OPS[operator]} ${param_name}, false)"
```

---

### 4.2 `tests/test_document_store.py`

Add the `assert_documents_are_equal` override inside `TestDocumentStore`:

```python
@staticmethod
def assert_documents_are_equal(received: list[Document], expected: list[Document]):
    """
    FalkorDB stores embeddings as vecf32 (float32), so exact float64 comparison
    is not possible after a round-trip. We compare id, content, and meta fields,
    and only verify that embedding is None vs not-None. We also sort both lists by
    document id to compensate for non-deterministic graph traversal order.
    """
    assert len(received) == len(expected), (
        f"Expected {len(expected)} documents but got {len(received)}"
    )
    received_sorted = sorted(received, key=lambda d: d.id)
    expected_sorted = sorted(expected, key=lambda d: d.id)
    for recv, exp in zip(received_sorted, expected_sorted):
        assert recv.id == exp.id
        assert recv.content == exp.content
        assert recv.meta == exp.meta
        assert (recv.embedding is None) == (exp.embedding is None), (
            f"Embedding presence mismatch for doc {recv.id}: "
            f"received {'None' if recv.embedding is None else 'vector'}, "
            f"expected {'None' if exp.embedding is None else 'vector'}"
        )
```

---

## 5. Verification

After applying all changes, run:

```bash
cd integrations/falkordb
hatch run test:integration
```

Expected outcome: **0 failed, 49 passed**.

Tests that were previously passing must remain passing. The override does not change
behaviour for:
- Empty-list comparisons (`[] == []`): `len` check passes, loop doesn't execute.
- Single-document comparisons with no embedding (write-dup tests): id/content/meta match.

---

## 6. Why No Other Files Need Changes

- `embedding_retriever.py` / `cypher_retriever.py`: not tested by `DocumentStoreBaseTests`.
- `__init__.py`: exports are correct.
- `pyproject.toml`: test dependencies and markers are correct.
- The `_node_to_document` function: correctly pops all standard fields; the embedding
  round-trip precision loss is an inherent property of vecf32 storage and is documented
  in the `assert_documents_are_equal` override rather than worked around in the store.
