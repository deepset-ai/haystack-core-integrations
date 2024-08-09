from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.openvino import OpenVINOChatGenerator

generator = OpenVINOChatGenerator(model="/home/ethan/intel/haystack/Phi-3-mini-4k-instruct-ov-int4")
generator.warm_up()
messages = [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
print(generator.run(messages))
