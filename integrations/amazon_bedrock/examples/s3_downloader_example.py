# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enabled access to the selected S3 bucket
# 3) `S3_DOWNLOADER_BUCKET` environment variable should be set to the name of the S3 bucket.

# The example shows how to use the S3Downloader component in a query pipeline to download files from an S3 bucket.
# To run this example, set the file_name in docs to your files in the S3 bucket.
# The files are then downloaded, converted to images and used to answer a question.


from pathlib import Path
from uuid import uuid4

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters.image import DocumentToImageContent
from haystack.components.routers import DocumentTypeRouter
from haystack.dataclasses import Document

from haystack_integrations.components.downloaders.s3.s3_downloader import S3Downloader
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

docs = [
    Document(meta={"file_id": str(uuid4()), "file_name": "text-sample.txt"}),
    Document(meta={"file_id": str(uuid4()), "file_name": "document-sample.pdf", "page_number": 1}),
]

chat_prompt_builder = ChatPromptBuilder(
    required_variables=["question"],
    template="""{% message role="system" %}
You are a friendly assistant that answers questions based on provided documents and images.
{% endmessage %}

{%- message role="user" -%}
Only provide an answer to the question using the images and text passages provided.

These are the text-only documents:
{%- if documents|length > 0 %}
{%- for doc in documents %}
Text Document [{{ loop.index }}] :
{{ doc.content }}
{% endfor -%}
{%- else %}
No relevant text documents were found.
{% endif %}
End of text documents.

Question: {{ question }}
Answer:

Images:
{%- if image_contents|length > 0 %}
{%- for img in image_contents -%}
  {{ img | templatize_part }}
{%- endfor -%}
{% endif %}
{%- endmessage -%}
""",
)

pipe = Pipeline()
pipe.add_component(
    "s3_downloader", S3Downloader(file_root_path=str(Path.cwd() / "s3_downloads"), file_extensions=[".pdf"])
)
pipe.add_component(
    "doc_type_router", DocumentTypeRouter(file_path_meta_field="file_path", mime_types=["application/pdf"])
)
pipe.add_component("doc_to_image", DocumentToImageContent(detail="auto"))
pipe.add_component("chat_prompt_builder", chat_prompt_builder)
pipe.add_component("llm", AmazonBedrockChatGenerator(model="anthropic.claude-3-haiku-20240307-v1:0"))

pipe.connect("s3_downloader.documents", "doc_type_router.documents")
pipe.connect("doc_type_router.application/pdf", "doc_to_image.documents")
pipe.connect("doc_to_image.image_contents", "chat_prompt_builder.image_contents")
pipe.connect("s3_downloader.documents", "chat_prompt_builder.documents")
pipe.connect("chat_prompt_builder.prompt", "llm.messages")

result = pipe.run(
    data={
        "s3_downloader": {"documents": docs},
        "chat_prompt_builder": {"question": "What is the main topic of the document?"},
    }
)
