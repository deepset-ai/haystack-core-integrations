# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enable access to the selected models for prompt routing in Amazon Bedrock

# You can find the ARN of a prompt router in the AWS console.
# For more information on prompt routing, see the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html).

from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

# we can skip the model parameter because we are using prompt routing
generator = AmazonBedrockChatGenerator(
    prompt_router_arn=Secret.from_token("your-prompt-router-arn"),
)

system_prompt = """
You are a helpful assistant that helps users learn more about AWS services.
Your audience is engineers with a decent technical background.
Be very concise and specific in your answers, keeping them short.
You may use technical terms, jargon, and abbreviations that are common among practitioners.
"""

messages = [
    ChatMessage.from_system(system_prompt),
    ChatMessage.from_user("Which service should I use to train custom Machine Learning models?"),
]

results = generator.run(messages)
