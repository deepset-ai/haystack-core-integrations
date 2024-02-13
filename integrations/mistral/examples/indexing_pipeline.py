from haystack import Document    
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder

doc = Document(content="I love pizza!")

document_embedder = MistralDocumentEmbedder()

result = document_embedder.run([doc])
print(result['documents'][0].embedding)