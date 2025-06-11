# To run this example, you will need to
# 1) set `GOOGLE_API_KEY` environment variable
# 2) install the google_genai_haystack integration: pip install google-genai-haystack
# Note: if you change the model, update the model-specific inference parameters.


from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

generator = GoogleGenAIChatGenerator(
    model="gemini-2.0-flash",
    # model-specific inference parameters
    generation_kwargs={
        "temperature": 0.7,
    },
)

system_prompt = """
You are a helpful assistant that helps users learn more about Google Cloud services.
Your audience is engineers with a decent technical background.
Be very concise and specific in your answers, keeping them short.
You may use technical terms, jargon, and abbreviations that are common among practitioners.
"""

messages = [
    ChatMessage.from_system(system_prompt),
    ChatMessage.from_user("Which service should I use to train custom Machine Learning models?"),
]

results = generator.run(messages)
print(results["replies"][0].text)
