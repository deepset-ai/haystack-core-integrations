# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional, Union

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
            result_str += document.content or "" + "\n"
        else:
            result_str += f"File Content for {document.meta['path']}\n\n"
            result_str += document.content or ""

    if len(result_str) > max_length:
        result_str = result_str[:max_length] + "...(large file can't be fully displayed)"

    return result_str


def serialize_handlers(
    serialized: Dict[str, Any],
    outputs_to_state: Optional[Dict[str, Dict[str, Union[str, Callable]]]],
    outputs_to_string: Optional[Dict[str, Union[str, Callable[[Any], str]]]],
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
                if not callable(config["handler"]):
                    msg = f"Handler for outputs_to_state[{key}] is not a callable"
                    raise ValueError(msg)
                serialized_config["handler"] = serialize_callable(config["handler"])
            serialized_outputs[key] = serialized_config
        serialized["outputs_to_state"] = serialized_outputs

    if outputs_to_string is not None and "handler" in outputs_to_string:
        serialized_string = outputs_to_string.copy()
        if not callable(outputs_to_string["handler"]):
            msg = "Handler for outputs_to_string is not a callable"
            raise ValueError(msg)
        serialized_string["handler"] = serialize_callable(outputs_to_string["handler"])
        serialized["outputs_to_string"] = serialized_string


def deserialize_handlers(data: Dict[str, Any]) -> None:
    """
    Deserializes callable handlers in outputs_to_state and outputs_to_string.

    :param data: The dictionary containing serialized handlers to deserialize
    """
    if data.get("outputs_to_state"):
        for config in data["outputs_to_state"].values():
            if "handler" in config:
                config["handler"] = deserialize_callable(config["handler"])

    if "outputs_to_string" in data and data["outputs_to_string"] and "handler" in data["outputs_to_string"]:
        data["outputs_to_string"]["handler"] = deserialize_callable(data["outputs_to_string"]["handler"])
