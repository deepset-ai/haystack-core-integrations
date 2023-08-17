import os
from pathlib import Path

from haystack.preview import Pipeline
from haystack.preview.components import TextFileToDocument, DocumentWriter

from chroma_haystack import ChromaDocumentStore
from chroma_haystack.retriever import ChromaDenseRetriever

HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir("data")]

document_store = ChromaDocumentStore()

indexing = Pipeline()
indexing.add_document_store("chroma", document_store)
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(), document_store="chroma")
indexing.connect("converter", "writer")
indexing.run({"converter": {"paths": file_paths}})

querying = Pipeline()
querying.add_document_store("chroma", document_store)
querying.add_component("retriever", ChromaDenseRetriever(), document_store="chroma")
results = querying.run({"retriever": {"queries": ["Variable declarations"], "top_k": 3}})

for d in results["retriever"][0]:
    print(d.metadata, d.score)
