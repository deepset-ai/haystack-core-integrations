from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.openvino import OpenVINOChatGenerator

generator = OpenVINOChatGenerator(model="microsoft/Phi-3-mini-4k-instruct")
generator.warm_up()
messages = [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
