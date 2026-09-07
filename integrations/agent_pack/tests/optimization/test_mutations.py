import json
from copy import deepcopy

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from pydantic import ValidationError

from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    MutationOperation,
    apply_mutation,
    rebuild_agent,
)


def reference_agent():
    """Create a serializable Agent fixture with scalar and container configuration."""
    return Agent(chat_generator=MockChatGenerator(model="reference"), system_prompt="reference prompt")


def test_multi_operation_mutation_can_edit_any_agent_configuration_area():
    """One decision may jointly edit model, prompt, loop bounds, and nested generator kwargs."""
    reference = reference_agent()
    mutation = AgentMutation(
        operations=(
            MutationOperation(
                op="set", path="/init_parameters/chat_generator/init_parameters/model", value="candidate"
            ),
            MutationOperation(op="set", path="/init_parameters/system_prompt", value="short prompt"),
            MutationOperation(op="set", path="/init_parameters/max_agent_steps", value=4),
            MutationOperation(op="create_object", path="/init_parameters/chat_generator/init_parameters/meta"),
            MutationOperation(
                op="set",
                path="/init_parameters/chat_generator/init_parameters/meta/temperature",
                value=0.2,
            ),
        )
    )

    serialized = apply_mutation(serialized_agent=reference.to_dict(), mutation=mutation)
    candidate = rebuild_agent(serialized_agent=serialized)

    assert candidate.chat_generator.model == "candidate"
    assert candidate.system_prompt == "short prompt"
    assert candidate.max_agent_steps == 4
    assert serialized["init_parameters"]["chat_generator"]["init_parameters"]["meta"] == {"temperature": 0.2}
    assert reference.chat_generator.model == "reference"
    assert reference.system_prompt == "reference prompt"


def test_container_copy_remove_array_append_and_escaped_paths():
    """The mutation language supports arbitrary JSON trees without embedding free-form JSON in the schema."""
    original = {"a/b": {"~source": [1, 2]}, "target": None}
    mutation = AgentMutation(
        operations=(
            MutationOperation(op="copy", from_path="/a~1b/~0source", path="/target"),
            MutationOperation(op="remove", path="/target/0"),
            MutationOperation(op="set", path="/target/-", value=3),
            MutationOperation(op="create_array", path="/empty"),
        )
    )

    changed = apply_mutation(serialized_agent=original, mutation=mutation)

    assert changed == {"a/b": {"~source": [1, 2]}, "target": [2, 3], "empty": []}
    assert original == {"a/b": {"~source": [1, 2]}, "target": None}


def test_invalid_path_and_invalid_deserialized_agent_are_explained():
    """Path and Haystack deserialization failures are surfaced as actionable errors."""
    with pytest.raises(ValueError, match="RFC 6901"):
        apply_mutation(
            serialized_agent={"value": 1},
            mutation=AgentMutation(operations=(MutationOperation(op="set", path="value", value=2),)),
        )
    reference = reference_agent()
    unsupported = AgentMutation(
        operations=(MutationOperation(op="set", path="/init_parameters/unsupported_parameter", value=True),)
    )
    with pytest.raises(ValueError, match="could not be rebuilt"):
        rebuild_agent(serialized_agent=apply_mutation(serialized_agent=reference.to_dict(), mutation=unsupported))
    mutated_class = AgentMutation(operations=(MutationOperation(op="set", path="/type", value="haystack.NoSuchAgent"),))
    with pytest.raises(ValueError, match="could not be rebuilt"):
        rebuild_agent(serialized_agent=apply_mutation(serialized_agent=reference.to_dict(), mutation=mutated_class))


def test_structured_models_forbid_unrecognized_operations_and_fields():
    """Provider structured output cannot silently invent mutation syntax."""
    with pytest.raises(ValidationError):
        AgentMutation.model_validate({"operations": [{"op": "merge", "path": "/x", "value": {}}]})
    with pytest.raises(ValidationError):
        AgentMutation.model_validate({"operations": [{"op": "remove", "path": "/x", "extra": True}]})


def test_mutation_fingerprint_is_stable_and_value_sensitive():
    """Invalid candidates can be deduplicated before a resulting configuration exists."""
    first = AgentMutation(operations=(MutationOperation(op="set", path="/x", value=1),))
    same = AgentMutation.model_validate(deepcopy(first.model_dump()))
    other = AgentMutation(operations=(MutationOperation(op="set", path="/x", value=2),))
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != other.fingerprint()


def test_set_json_replaces_a_component_with_a_differently_shaped_one():
    """Retuning a component is one operation; replacing it with a pipeline has to be one too."""
    reference = reference_agent()
    pipeline_tool = {
        "type": "haystack.tools.PipelineTool",
        "data": {"name": "search_documents", "description": "Retrieve then rerank.", "pipeline": {"components": {}}},
    }
    mutation = AgentMutation(
        operations=(MutationOperation(op="set_json", path="/init_parameters/tools/0", value=json.dumps(pipeline_tool)),)
    )

    changed = apply_mutation(serialized_agent=reference.to_dict(), mutation=mutation)

    assert changed["init_parameters"]["tools"][0] == pipeline_tool
    assert changed["init_parameters"]["tools"][0]["data"]["name"] == "search_documents"


def test_set_json_rejects_text_that_is_not_json_and_values_that_are_not_text():
    """A subtree written as text has to be text, and has to parse, or the failure is silent corruption."""
    with pytest.raises(ValidationError, match="JSON string"):
        MutationOperation(op="set_json", path="/x", value=3)

    broken = AgentMutation(operations=(MutationOperation(op="set_json", path="/x", value="{not json"),))
    with pytest.raises(ValueError, match="invalid JSON"):
        apply_mutation(serialized_agent={"x": 1}, mutation=broken)
