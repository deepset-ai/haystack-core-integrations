# To run this example, you will need to:
# 1) Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` environment variables
# 2) Place a document image named `invoice.png` in the same directory as this script
#
# This example demonstrates structural analysis using AWS Textract's AnalyzeDocument API.
# Setting `feature_types` enables extraction of tables, forms, and layout information
# in addition to plain text.

from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

converter = AmazonTextractConverter(feature_types=["TABLES", "FORMS"])

results = converter.run(sources=["invoice.png"])

for doc in results["documents"]:
    print(f"--- {doc.meta.get('file_path', 'unknown')} ---")
    print(doc.content)
    print()

raw = results["raw_textract_response"][0]
table_blocks = [b for b in raw.get("Blocks", []) if b.get("BlockType") == "TABLE"]
print(f"Tables found: {len(table_blocks)}")

form_blocks = [b for b in raw.get("Blocks", []) if b.get("BlockType") == "KEY_VALUE_SET"]
print(f"Key-value pairs found: {len(form_blocks)}")
