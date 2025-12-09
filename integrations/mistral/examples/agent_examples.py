from haystack.dataclasses import ChatMessage
from haystack_integrations.components.agents import MistralAgentGenerator


generator = MistralAgentGenerator(agent_id="ag-12345678-abcd-efgh-ijkl-mnopqrstuvwx")
messages = [ChatMessage.from_user("Hello! What can you help me with?")]
result = generator.run(messages)
print(result["replies"][0].text)