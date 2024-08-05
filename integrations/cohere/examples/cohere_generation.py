# This example demonstrates a corrective Haystack pipeline with Cohere LLM integration that runs until the
# generated output satisfies a strict JSON schema.
#
# The pipeline includes the following components:
#  - BranchJoiner: https://docs.haystack.deepset.ai/reference/joiners-api#branchjoiner
#  - JsonSchemaValidator: https://docs.haystack.deepset.ai/reference/validators-api#jsonschemavalidator
#
# The pipeline workflow:
# 1. Receives a user message requesting to create a JSON object from "Peter Parker" aka Superman.
# 2. Processes the message through components to generate a response using Cohere command-r model.
# 3. Validates the generated response against a predefined JSON schema for person data.
# 4. If the response does not meet the schema, the JsonSchemaValidator provides details on how to correct the errors.
# 4a. The pipeline loops back, using the error information to generate a new JSON object until it satisfies the schema.
# 5. If the response is validated against the schema, outputs the validated JSON object.

from typing import List

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.cohere import CohereChatGenerator

# Defines a JSON schema for validating a person's data. The schema specifies that a valid object must
# have first_name, last_name, and nationality properties, with specific constraints on their values.
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
