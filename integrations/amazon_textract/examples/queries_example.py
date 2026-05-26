# To run this example, you will need to:
# 1) Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` environment variables
# 2) Place a document image named `medical_form.png` in the same directory as this script
#
# This example demonstrates natural-language queries using AWS Textract.
# The QUERIES feature type is enabled automatically when you pass the `queries`
# parameter at runtime. Textract will attempt to find answers to each question
# in the document.

from haystack_integrations.components.converters.amazon_textract import AmazonTextractConverter

converter = AmazonTextractConverter()

results = converter.run(
    sources=["medical_form.png"],
    queries=["What is the patient name?", "What is the date of birth?", "What is the diagnosis?"],
)

for doc in results["documents"]:
    print("--- Extracted text ---")
    print(doc.content)
    print()

raw = results["raw_textract_response"][0]
query_blocks = [b for b in raw.get("Blocks", []) if b.get("BlockType") == "QUERY"]
for block in query_blocks:
    question = block.get("Query", {}).get("Text", "")
    print(f"Q: {question}")

query_result_blocks = [b for b in raw.get("Blocks", []) if b.get("BlockType") == "QUERY_RESULT"]
for block in query_result_blocks:
    answer = block.get("Text", "")
    confidence = block.get("Confidence", 0)
    print(f"A: {answer} (confidence: {confidence:.1f}%)")
