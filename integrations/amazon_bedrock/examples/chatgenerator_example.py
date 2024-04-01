# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enable access to the selected model in Amazon Bedrock
# Note: if you change the model, update the model-specific inference parameters.


from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

system_prompt = """
You are a helpful assistant that helps users learn more about AWS services.
Your audience is engineers with a decent technical background.
Be very concise and specific in your answers, keeping them short.
You may use technical terms, jargon, and abbreviations that are common among practitioners.
"""

generator = AmazonBedrockChatGenerator(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    # model-specific inference parameters
    generation_kwargs={
        "system": system_prompt,
        "max_tokens": 500,
        "temperature": 0.0,
    },
)
messages = [
    ChatMessage.from_user("Which service should I use to train custom Machine Learning models?"),
]

results = generator.run(messages)
results["replies"]
