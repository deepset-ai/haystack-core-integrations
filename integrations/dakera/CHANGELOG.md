# Changelog

## [integrations/dakera-v0.1.0]

- Initial release of the Dakera integration for Haystack.
- Memory integration: `DakeraMemoryStore` (sync + async) with `DakeraMemoryWriter` and
  `DakeraMemoryRetriever` for conversational, decay-weighted memory over the Dakera memory API
  (`ChatMessage`-based). All components support `run` and `run_async`.
- Document store: `DakeraDocumentStore` (sync + async) backed by the Dakera vector-namespace API.
- Adds `DakeraEmbeddingRetriever` (sync + async) for dense retrieval, with filter-policy support.
