# Currently, this example shows how to use the FastembedSparseDocumentEmbedder component to embed a list of documents.

# TODO: Once we have a proper SparseEmbeddingRetriever, we should replace this naive example with a more realistic one,
# involving indexing and retrieval of documents.

from haystack import Document

from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder

document_list = [
    Document(
        content="Oxidative stress generated within inflammatory joints can produce autoimmune phenomena and joint destruction. Radical species with oxidative activity, including reactive nitrogen species, represent mediators of inflammation and cartilage damage.",
        meta={
            "pubid": "25,445,628",
            "long_answer": "yes",
        },
    ),
    Document(
        content="Plasma levels of pancreatic polypeptide (PP) rise upon food intake. Although other pancreatic islet hormones, such as insulin and glucagon, have been extensively investigated, PP secretion and actions are still poorly understood.",
        meta={
            "pubid": "25,445,712",
            "long_answer": "yes",
        },
    ),
]

document_embedder = FastembedSparseDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(document_list)["documents"]

for doc in documents_with_embeddings:
    print(f"Document Text: {doc.content}")
    print(f"Document Sparse Embedding: {doc.sparse_embedding.to_dict()}")
