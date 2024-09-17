# In order to run this example, you will need to have an instance of Ollama running with the
# orca-mini model downloaded. We suggest you use the following commands to serve an orca-mini
# model from Ollama
#
# docker run -d -p 11434:11434 --name ollama ollama/ollama:latest
# docker exec ollama ollama pull orca-mini

from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.ollama import OllamaChatGenerator

messages = [
    ChatMessage.from_user("What's Natural Language Processing?"),
    ChatMessage.from_assistant(
        "Natural Language Processing (NLP) is a field of computer science and artificial "
        "intelligence concerned with the interaction between computers and human language"
    ),
    ChatMessage.from_user("How do I get started?"),
]
client = OllamaChatGenerator(model="orca-mini", timeout=45, url="http://localhost:11434")

response = client.run(messages, generation_kwargs={"temperature": 0.2})

print(response["replies"])
#
# [
#     ChatMessage(
#         content="Natural Language Processing (NLP) is a broad field of computer science and artificial intelligence "
#                 "that involves the interaction between computers and human language. To get started in NLP, "
#                 "you can start by learning about the different techniques and tools used in NLP such as machine "
#                 "learning algorithms, deep learning frameworks, and natural language processing libraries. You can "
#                 "also learn about the applications of NLP in various fields such as chatbots, sentiment analysis, "
#                 "speech recognition, and text classification. Additionally, you can explore the available resources "
#                 "such as online courses, tutorials, and books on NLP to gain a deeper understanding of the field.",
#         role=<ChatRole.ASSISTANT: 'assistant'>,
#         name=None,
#         meta={
#             "model": "orca-mini",
#             "created_at": "2024-01-08T15:35:23.378609793Z",
#             "done": True,
#             "total_duration": 20026330217,
#             "load_duration": 1540167,
#             "prompt_eval_count": 99,
#             "prompt_eval_duration": 8486609000,
#             "eval_count": 124,
#             "eval_duration": 11532988000,
#         },
#     )
# ]
