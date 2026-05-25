# To run this example, you will need to:
# 1) Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` environment variables
# 2) Place an image or single-page PDF named `document.png` in the same directory as this script
#
# This example demonstrates basic text extraction from a document image using
# AWS Textract's DetectDocumentText API.

from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

converter = AmazonTextractConverter()

results = converter.run(sources=["document.png"])

for doc in results["documents"]:
    print(f"--- {doc.meta.get('file_path', 'unknown')} (pages: {doc.meta.get('page_count')}) ---")
    print(doc.content)
    print()
