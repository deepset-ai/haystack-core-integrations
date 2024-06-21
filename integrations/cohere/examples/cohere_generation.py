from typing import List

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.cohere import CohereChatGenerator

person_schema = {
    "type": "object",
    "properties": {
        "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
        "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
    },
    "required": ["first_name", "last_name", "nationality"],
}

# Initialize a pipeline
pipe = Pipeline()

# Add components to the pipeline
pipe.add_component("joiner", BranchJoiner(List[ChatMessage]))
pipe.add_component("fc_llm", CohereChatGenerator(model="command-r"))
pipe.add_component("validator", JsonSchemaValidator(json_schema=person_schema))
pipe.add_component("adapter", OutputAdapter("{{chat_message}}", List[ChatMessage])),
# And connect them
pipe.connect("adapter", "joiner")
pipe.connect("joiner", "fc_llm")
pipe.connect("fc_llm.replies", "validator.messages")
pipe.connect("validator.validation_error", "joiner")

result = pipe.run(data={"adapter": {"chat_message": [ChatMessage.from_user("Create json from Peter Parker")]}})

print(result["validator"]["validated"])  # noqa: T201
