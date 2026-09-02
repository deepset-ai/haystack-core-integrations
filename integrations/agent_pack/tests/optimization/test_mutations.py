from copy import deepcopy

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from pydantic import ValidationError

from haystack_integrations.agent_pack.optimization import (
    AgentMutation,
    MutationOperation,
    apply_mutation,
    materialize_mutation,
)
from haystack_integrations.agent_pack.optimization.mutations import mutation_fingerprint


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

    candidate, serialized = materialize_mutation(reference=reference, mutation=mutation)

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
    with pytest.raises(ValueError, match="could not be rebuilt"):
        materialize_mutation(
            reference=reference_agent(),
            mutation=AgentMutation(
                operations=(MutationOperation(op="set", path="/init_parameters/unsupported_parameter", value=True),)
            ),
        )


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
    assert mutation_fingerprint(mutation=first) == mutation_fingerprint(mutation=same)
    assert mutation_fingerprint(mutation=first) != mutation_fingerprint(mutation=other)
