# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enable access to the selected model in Amazon Bedrock
# Note: if you change the model, update the model-specific inference parameters.


from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

generator = AmazonBedrockChatGenerator(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    # model-specific inference parameters
    generation_kwargs={
        "max_tokens": 500,
        "temperature": 0.0,
    },
)

system_prompt = """
You are a helpful assistant that helps users learn more about AWS services.
Your audience is engineers with a decent technical background.
Be very concise and specific in your answers, keeping them short.
You may use technical terms, jargon, and abbreviations that are common among practitioners.
"""

# Even though Anthropic Claud models support only messages with `user` and `assistant` roles,
# internal handling converts message with `system` role into `system` inference parameter for Claude
# which allows for more portablability of code across generators
messages = [
    ChatMessage.from_system(system_prompt),
    ChatMessage.from_user("Which service should I use to train custom Machine Learning models?"),
]

results = generator.run(messages)
results["replies"]
