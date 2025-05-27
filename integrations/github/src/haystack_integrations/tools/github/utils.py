# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Union

from haystack import Document
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


def message_handler(documents: List[Document], max_length: int = 150_000) -> str:
    """
    Handles the tool output before conversion to ChatMessage.

    :param documents: List of Document objects
    :param max_length: Maximum number of characters of the result string
    :returns:
        String representation of the documents.
    """
    result_str = ""
    for document in documents:
        if document.meta["type"] in ["file", "dir", "error"]:
            result_str += document.content + "\n"
        else:
            result_str += f"File Content for {document.meta['path']}\n\n"
            result_str += document.content

    if len(result_str) > max_length:
        result_str = result_str[:max_length] + "...(large file can't be fully displayed)"

    return result_str


def serialize_handlers(
    serialized: Dict[str, Any],
    outputs_to_state: Dict[str, Dict[str, Union[str, Callable]]],
    outputs_to_string: Dict[str, Union[str, Callable[[Any], str]]],
) -> None:
    """
    Serializes callable handlers in outputs_to_state and outputs_to_string.

    :param serialized: The dictionary to update with serialized handlers
    :param outputs_to_state: Dictionary containing state output configurations
    :param outputs_to_string: Dictionary containing string output configuration
    """
    if outputs_to_state is not None:
        serialized_outputs = {}
        for key, config in outputs_to_state.items():
            serialized_config = config.copy()
            if "handler" in config:
                serialized_config["handler"] = serialize_callable(config["handler"])
            serialized_outputs[key] = serialized_config
        serialized["init_parameters"]["outputs_to_state"] = serialized_outputs

    if outputs_to_string is not None and outputs_to_string.get("handler") is not None:
        serialized["init_parameters"]["outputs_to_string"] = {
            **outputs_to_string,
            "handler": serialize_callable(outputs_to_string["handler"]),
        }


def deserialize_handlers(data: Dict[str, Any]) -> None:
    """
    Deserializes callable handlers in outputs_to_state and outputs_to_string.

    :param data: The dictionary containing serialized handlers to deserialize
    """
    if "outputs_to_state" in data["init_parameters"] and data["init_parameters"]["outputs_to_state"]:
        deserialized_outputs = {}
        for key, config in data["init_parameters"]["outputs_to_state"].items():
            deserialized_config = config.copy()
            if "handler" in config:
                deserialized_config["handler"] = deserialize_callable(config["handler"])
            deserialized_outputs[key] = deserialized_config
        data["init_parameters"]["outputs_to_state"] = deserialized_outputs

    if (
        data["init_parameters"].get("outputs_to_string") is not None
        and data["init_parameters"]["outputs_to_string"].get("handler") is not None
    ):
        data["init_parameters"]["outputs_to_string"]["handler"] = deserialize_callable(
            data["init_parameters"]["outputs_to_string"]["handler"]
        )
