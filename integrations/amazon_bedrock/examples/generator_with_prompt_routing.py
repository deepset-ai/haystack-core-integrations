# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enable access to the selected models for prompt routing in Amazon Bedrock

# To see how to setup prompt router, you can refer to
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/create_prompt_router.html

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

# You can use either foundation-model ARNs OR inference-profile ARNs.
NOVA_PRO = "nove_pro_model_arn"
NOVA_LITE = "nova_lite_model_arn"

ROUTER_NAME = "custom_nova_router"

RESPONSE_QUALITY_DIFF = 10.0  # interpreted as a quality delta threshold
PromptRouterConfig = {
    "promptRouterName": ROUTER_NAME,
    "models": [{"modelArn": NOVA_PRO}, {"modelArn": NOVA_LITE}],
    "description": "Demo prompt router that picks between Nova Pro and Nova Lite.",
    "routingCriteria": {"responseQualityDifference": RESPONSE_QUALITY_DIFF},
    "fallbackModel": {"modelArn": NOVA_LITE},
    "tags": [{"key": "env", "value": "demo"}],
}

# we can skip the model parameter because we are using prompt routing
generator = AmazonBedrockChatGenerator(
    prompt_router_config=PromptRouterConfig,
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
