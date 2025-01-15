import os
from haystack import Document
from haystack.utils import Secret
from haystack_integrations.components.rankers.amazon_bedrock import BedrockRanker
from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)


def main():
    # Set up AWS credentials
    # You can also set these as environment variables
    aws_profile_name = os.environ.get("AWS_PROFILE") or "default"
    aws_region_name = os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"

    try:
        # Initialize the BedrockRanker with AWS credentials
        ranker = BedrockRanker(
            model="cohere.rerank-v3-5:0",
            top_k=2,
            aws_profile_name=Secret.from_token(aws_profile_name),
            aws_region_name=Secret.from_token(aws_region_name),
        )

        # Create some sample documents
        docs = [
            Document(content="Paris is the capital of France."),
            Document(content="Berlin is the capital of Germany."),
            Document(content="London is the capital of the United Kingdom."),
            Document(content="Rome is the capital of Italy."),
        ]

        # Define a query
        query = "What is the capital of Germany?"

        # Run the ranker
        output = ranker.run(query=query, documents=docs)

        # Print the results
        print("Query:", query)
        print("\nRanked Documents:")
        for i, doc in enumerate(output["documents"], 1):
            print(f"{i}. {doc.content} (Score: {doc.score:.4f})")

    except AmazonBedrockConfigurationError as config_error:
        print(f"Configuration Error: {config_error}")
    except AmazonBedrockInferenceError as inference_error:
        print(f"Inference Error: {inference_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
