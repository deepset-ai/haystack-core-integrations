# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Advanced pipeline example combining S3 Downloader with Amazon Bedrock components.

This example demonstrates how to create a complete document processing pipeline
that downloads files from S3 and processes them using Bedrock models.
"""

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.downloader import S3Downloader


def create_s3_to_bedrock_pipeline():
    """Create a pipeline that downloads from S3 and processes with Bedrock components."""
    
    # Create document store
    document_store = InMemoryDocumentStore()
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add S3 downloader (shares AWS credentials with Bedrock components)
    pipeline.add_component("s3_downloader", S3Downloader())
    
    # Add document processing components
    pipeline.add_component("converter", TextFileToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())
    pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=3))
    
    # Add Bedrock embedder (would use same AWS credentials)
    try:
        from haystack_integrations.components.embedders.amazon_bedrock import AmazonBedrockTextEmbedder
        embedder = AmazonBedrockTextEmbedder(model="amazon.titan-embed-text-v1")
        pipeline.add_component("embedder", embedder)
        
        # Add writer to store documents with embeddings
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        
        # Connect components
        pipeline.connect("s3_downloader.content", "converter.sources")
        pipeline.connect("converter.documents", "cleaner.documents")
        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "embedder.documents")
        pipeline.connect("embedder.documents", "writer.documents")
        
        print("✓ Full S3 + Bedrock embeddings pipeline created")
        return pipeline, document_store, True
        
    except ImportError:
        # Fallback pipeline without Bedrock embedder
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        
        pipeline.connect("s3_downloader.content", "converter.sources")
        pipeline.connect("converter.documents", "cleaner.documents")
        pipeline.connect("cleaner.documents", "splitter.documents")
        pipeline.connect("splitter.documents", "writer.documents")
        
        print("✓ S3 document processing pipeline created (without Bedrock embeddings)")
        return pipeline, document_store, False


def create_s3_to_bedrock_generator_pipeline():
    """Create a pipeline for Q&A using S3 documents and Bedrock generator."""
    
    pipeline = Pipeline()
    
    # Add S3 downloader
    pipeline.add_component("s3_downloader", S3Downloader())
    pipeline.add_component("converter", TextFileToDocument())
    
    try:
        from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator
        
        # Add Bedrock generator
        generator = AmazonBedrockGenerator(model="anthropic.claude-3-sonnet-20240229-v1:0")
        pipeline.add_component("generator", generator)
        
        # Connect components
        pipeline.connect("s3_downloader.content", "converter.sources")
        pipeline.connect("converter.documents", "generator.documents")
        
        print("✓ S3 + Bedrock generator pipeline created")
        return pipeline, True
        
    except ImportError:
        print("⚠ Bedrock generator not available - basic pipeline created")
        return pipeline, False


def main():
    """Demonstrate S3 downloader integration with Amazon Bedrock ecosystem."""
    
    print("S3 Downloader + Amazon Bedrock Integration Examples")
    print("=" * 54)
    
    # Example 1: Document processing with embeddings
    print("\n--- Example 1: S3 to Bedrock Embeddings Pipeline ---")
    try:
        pipeline, document_store, has_bedrock = create_s3_to_bedrock_pipeline()
        
        print(f"Pipeline components: {len(pipeline.graph.nodes)}")
        if has_bedrock:
            print("Components: S3Downloader -> TextFileToDocument -> DocumentCleaner -> DocumentSplitter -> BedrockEmbedder -> DocumentWriter")
        else:
            print("Components: S3Downloader -> TextFileToDocument -> DocumentCleaner -> DocumentSplitter -> DocumentWriter")
        
        # Show how to run the pipeline
        print("\nTo run this pipeline:")
        print("result = pipeline.run({")
        print('    "s3_downloader": {"url": "s3://my-bucket/document.txt"}')
        print("})")
        
        if has_bedrock:
            print("\nThis pipeline will:")
            print("1. Download document from S3 using shared AWS credentials")
            print("2. Convert to Haystack document format")
            print("3. Clean and split the document")
            print("4. Generate embeddings using Bedrock")
            print("5. Store in document store for retrieval")
        
    except Exception as e:
        print(f"✗ Could not create embeddings pipeline: {e}")
    
    # Example 2: Q&A with Bedrock generator
    print("\n--- Example 2: S3 + Bedrock Generator Pipeline ---")
    try:
        qa_pipeline, has_generator = create_s3_to_bedrock_generator_pipeline()
        
        if has_generator:
            print("✓ Q&A pipeline created successfully")
            print("Components: S3Downloader -> TextFileToDocument -> BedrockGenerator")
            
            print("\nTo run this pipeline:")
            print("result = qa_pipeline.run({")
            print('    "s3_downloader": {"url": "s3://my-docs/manual.pdf"},')
            print('    "generator": {"query": "What are the key features?"}')
            print("})")
        
    except Exception as e:
        print(f"✗ Could not create generator pipeline: {e}")
    
    # Example 3: Multi-document processing
    print("\n--- Example 3: Multi-Document S3 Processing ---")
    try:
        multi_pipeline = Pipeline()
        
        # Add multiple S3 downloaders for different files
        for i in range(3):
            multi_pipeline.add_component(f"s3_downloader_{i+1}", S3Downloader())
            multi_pipeline.add_component(f"converter_{i+1}", TextFileToDocument())
        
        # Add document joiner
        from haystack.components.joiners import DocumentJoiner
        multi_pipeline.add_component("joiner", DocumentJoiner())
        
        # Connect components
        for i in range(3):
            multi_pipeline.connect(f"s3_downloader_{i+1}.content", f"converter_{i+1}.sources")
            multi_pipeline.connect(f"converter_{i+1}.documents", "joiner.documents")
        
        print("✓ Multi-document S3 processing pipeline created")
        print(f"Components: {len(multi_pipeline.graph.nodes)}")
        
        print("\nTo run this pipeline:")
        print("result = multi_pipeline.run({")
        print('    "s3_downloader_1": {"url": "s3://bucket/doc1.txt"},')
        print('    "s3_downloader_2": {"url": "s3://bucket/doc2.txt"},')
        print('    "s3_downloader_3": {"url": "s3://bucket/doc3.txt"}')
        print("})")
        
    except ImportError as e:
        print(f"✗ Could not create multi-document pipeline: {e}")
    
    # Example 4: Shared AWS credentials configuration
    print("\n--- Example 4: Shared AWS Configuration ---")
    
    print("Benefits of S3Downloader in Amazon Bedrock integration:")
    print("1. Shared AWS credentials across all components")
    print("2. Consistent boto3 session management")
    print("3. Same authentication patterns (IAM roles, profiles, env vars)")
    print("4. Unified error handling for AWS-related issues")
    print("5. Compatible SSL and endpoint configurations")
    
    # Example configuration showing consistency
    print("\nExample shared configuration:")
    print("""
from haystack import Secret
from haystack_integrations.components.downloader import S3Downloader
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

# Shared AWS configuration
aws_config = {
    "aws_access_key_id": Secret.from_token("your_key"),
    "aws_secret_access_key": Secret.from_token("your_secret"),
    "aws_region": "us-west-2"
}

# All components use the same credentials
s3_downloader = S3Downloader(**aws_config)
bedrock_generator = AmazonBedrockGenerator(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    **aws_config
)
""")
    
    # Example 5: Error handling best practices
    print("\n--- Example 5: Error Handling Best Practices ---")
    
    print("Best practices when using S3Downloader with Bedrock:")
    print("1. Use warm_up() on all AWS components before pipeline execution")
    print("2. Handle AWS credential errors consistently")
    print("3. Validate S3 URLs before processing")
    print("4. Use appropriate AWS regions for both S3 and Bedrock")
    print("5. Monitor AWS service quotas and rate limits")
    
    print("\nExample error handling:")
    print("""
try:
    # Warm up all AWS components
    s3_downloader.warm_up()
    bedrock_generator.warm_up()
    
    # Run pipeline
    result = pipeline.run({
        "s3_downloader": {"url": "s3://my-bucket/file.txt"}
    })
    
except FileNotFoundError:
    print("S3 file not found")
except NoCredentialsError:
    print("AWS credentials not configured")
except ValueError as e:
    print(f"Invalid S3 URL: {e}")
except Exception as e:
    print(f"AWS service error: {e}")
""")
    
    print("\n--- S3 + Bedrock integration examples completed ---")


if __name__ == "__main__":
    main() 