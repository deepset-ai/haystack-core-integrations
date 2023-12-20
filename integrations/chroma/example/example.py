# Colab: https://colab.research.google.com/drive/1YpDetI8BRbObPDEVdfqUcwhEX9UUXP-m?usp=sharing
import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter

from chroma_haystack import ChromaDocumentStore
from chroma_haystack.retriever import ChromaQueryRetriever

HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir("data")]

# Chroma is used in-memory so we use the same instances in the two pipelines below
document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
indexing.run({"converter": {"sources": file_paths}})

querying = Pipeline()
querying.add_component("retriever", ChromaQueryRetriever(document_store))
results = querying.run({"retriever": {"queries": ["Variable declarations"], "top_k": 3}})

for d in results["retriever"]["documents"][0]:
    print(d.meta, d.score)
