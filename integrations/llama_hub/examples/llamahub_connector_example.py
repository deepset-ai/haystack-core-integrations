from haystack import Pipeline
from haystack_integrations.components.connectors.llamahub import LlamaHubConnector

def main():
    print("Initializing Haystack 2.0 Pipeline layout...")
    pipeline = Pipeline()
    
    # 1. Add our custom dynamic connector, identifying the target LlamaHub loader class
    pipeline.add_component(
        "loader", 
        LlamaHubConnector(reader_module="llama_index.readers.web",reader_class="SimpleWebPageReader")
    )
    
    print("Executing pipeline processing stream...")
    # 2. Run the pipeline passing parameters expected by SimpleWebPageReader's load_data method
    response = pipeline.run({
        "loader": {"urls": ["https://example.com"]}
    })
    
    # 3. Parse and print structural layout output results
    documents = response["loader"]["documents"]
    print(f"\nSuccess! Processed {len(documents)} document(s).")
    for doc in documents:
        print(f"--- Document Content Snippet ---")
        print(doc.content[:200] + "...")
        print(f"Metadata properties: {doc.meta}\n")

if __name__ == "__main__":
    main()