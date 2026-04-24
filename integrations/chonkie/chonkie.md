---
title: "Chonkie"
id: integrations-chonkie
description: "Chonkie integration for Haystack"
slug: "/integrations-chonkie"
---


## haystack_integrations.components.preprocessors.chonkie.recursive_chunker

### ChonkieRecursiveChunker

A Document Splitter that uses Chonkie's RecursiveChunker to split documents.

Usage::

```
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveChunker

chunker = ChonkieRecursiveChunker(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    tokenizer: str = "character",
    chunk_size: int = 2048,
    min_characters_per_chunk: int = 24,
    rules: Any = None,
) -> None
```

Initializes the ChonkieRecursiveChunker.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
- **chunk_size** (<code>int</code>) – The maximum size of each chunk.
- **min_characters_per_chunk** (<code>int</code>) – The minimum number of characters per chunk.
- **rules** (<code>Any</code>) – Custom rules for recursive chunking. If None, default rules are used.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Splits a list of documents into smaller chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieRecursiveChunker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieRecursiveChunker</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.semantic_chunker

### ChonkieSemanticChunker

A Document Splitter that uses Chonkie's SemanticChunker to split documents.

Usage::

```
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticChunker

chunker = ChonkieSemanticChunker(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    embedding_model: Any = "minishlab/potion-base-32M",
    threshold: float = 0.8,
    chunk_size: int = 2048,
    similarity_window: int = 3,
    min_sentences_per_chunk: int = 1,
    min_characters_per_sentence: int = 24,
    delim: Any = None,
    include_delim: str = "prev",
    skip_window: int = 0,
    filter_window: int = 5,
    filter_polyorder: int = 3,
    filter_tolerance: float = 0.2,
) -> None
```

Initializes the ChonkieSemanticChunker.

**Parameters:**

- **embedding_model** (<code>Any</code>) – The embedding model to use for semantic similarity.
- **threshold** (<code>float</code>) – The semantic similarity threshold.
- **chunk_size** (<code>int</code>) – The maximum size of each chunk.
- **similarity_window** (<code>int</code>) – The window size for similarity calculations.
- **min_sentences_per_chunk** (<code>int</code>) – The minimum number of sentences per chunk.
- **min_characters_per_sentence** (<code>int</code>) – The minimum number of characters per sentence.
- **delim** (<code>Any</code>) – Delimiters to use for splitting. If None, default delimiters are used.
- **include_delim** (<code>str</code>) – Whether to include the delimiter in the chunks.
- **skip_window** (<code>int</code>) – The skip window for similarity calculations.
- **filter_window** (<code>int</code>) – The filter window for similarity calculations.
- **filter_polyorder** (<code>int</code>) – The polynomial order for similarity filtering.
- **filter_tolerance** (<code>float</code>) – The tolerance for similarity filtering.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Splits a list of documents into smaller semantic chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieSemanticChunker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieSemanticChunker</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.sentence_chunker

### ChonkieSentenceChunker

A Document Splitter that uses Chonkie's SentenceChunker to split documents.

Usage::

```
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieSentenceChunker

chunker = ChonkieSentenceChunker(chunk_size=512)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    tokenizer: str = "character",
    chunk_size: int = 2048,
    chunk_overlap: int = 0,
    min_sentences_per_chunk: int = 1,
    min_characters_per_sentence: int = 12,
    approximate: bool = False,
    delim: Any = None,
    include_delim: str = "prev",
) -> None
```

Initializes the ChonkieSentenceChunker.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
- **chunk_size** (<code>int</code>) – The maximum size of each chunk.
- **chunk_overlap** (<code>int</code>) – The overlap between consecutive chunks.
- **min_sentences_per_chunk** (<code>int</code>) – The minimum number of sentences per chunk.
- **min_characters_per_sentence** (<code>int</code>) – The minimum number of characters per sentence.
- **approximate** (<code>bool</code>) – Whether to use approximate chunking.
- **delim** (<code>Any</code>) – Delimiters to use for splitting. If None, default delimiters are used.
- **include_delim** (<code>str</code>) – Whether to include the delimiter in the chunks ("prev" or "next").

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Splits a list of documents into smaller sentence-based chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieSentenceChunker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieSentenceChunker</code> – Deserialized component.

## haystack_integrations.components.preprocessors.chonkie.token_chunker

### ChonkieTokenChunker

A Document Splitter that uses Chonkie's TokenChunker to split documents.

Usage::

```
from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenChunker

chunker = ChonkieTokenChunker(chunk_size=512, chunk_overlap=50)
documents = [Document(content="Hello world. This is a test.")]
result = chunker.run(documents=documents)
print(result["documents"])
```

#### __init__

```python
__init__(
    tokenizer: str = "character", chunk_size: int = 2048, chunk_overlap: int = 0
) -> None
```

Initializes the ChonkieTokenChunker.

**Parameters:**

- **tokenizer** (<code>str</code>) – The tokenizer to use for chunking. Defaults to "character".
- **chunk_size** (<code>int</code>) – The maximum size of each chunk.
- **chunk_overlap** (<code>int</code>) – The overlap between consecutive chunks.

#### run

```python
run(documents: list[Document]) -> dict[str, Any]
```

Splits a list of documents into smaller token-based chunks.

**Parameters:**

- **documents** (<code>list\[Document\]</code>) – The list of documents to split.

**Returns:**

- <code>dict\[str, Any\]</code> – A dictionary with the "documents" key containing the list of chunks.

#### to_dict

```python
to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

**Returns:**

- <code>dict\[str, Any\]</code> – Dictionary with serialized data.

#### from_dict

```python
from_dict(data: dict[str, Any]) -> ChonkieTokenChunker
```

Deserializes the component from a dictionary.

**Parameters:**

- **data** (<code>dict\[str, Any\]</code>) – Dictionary to deserialize from.

**Returns:**

- <code>ChonkieTokenChunker</code> – Deserialized component.
