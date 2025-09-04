from typing import List, Optional

from haystack.dataclasses import ReasoningContent, StreamingChunk


def _process_reasoning_contents(chunks: List[StreamingChunk]) -> Optional[ReasoningContent]:
    """
    Process reasoning contents from a list of StreamingChunk objects into the Anthropic expected format.

    :param chunks: List of StreamingChunk objects potentially containing reasoning contents.

    :returns: List of Anthropic formatted reasoning content dictionaries
    """
    formatted_reasoning_contents = []
    current_index = None
    content_block_text = ""
    content_block_signature = None
    content_block_redacted_thinking = None
    content_block_index = None

    for chunk in chunks:
        if (delta := chunk.meta.get("delta")) is not None:
            if delta.get("type") == "thinking_delta" and delta.get("thinking") is not None:
                content_block_index = chunk.meta.get("index", None)
        if (content_block := chunk.meta.get("content_block")) is not None and content_block.get(
            "type"
        ) == "redacted_thinking":
            content_block_index = chunk.meta.get("index", None)

        # Start new group when index changes
        if current_index is not None and content_block_index != current_index:
            # Finalize current group
            if content_block_text:
                formatted_reasoning_contents.append(
                    {
                        "reasoning_content": {
                            "reasoning_text": {"text": content_block_text, "signature": content_block_signature},
                        }
                    }
                )
            if content_block_redacted_thinking:
                formatted_reasoning_contents.append(
                    {"reasoning_content": {"redacted_thinking": content_block_redacted_thinking}}
                )

            # Reset accumulators for new group
            content_block_text = ""
            content_block_signature = None
            content_block_redacted_thinking = None

        # Accumulate content for current index
        current_index = content_block_index
        if (delta := chunk.meta.get("delta")) is not None:
            if delta.get("type") == "thinking_delta" and delta.get("thinking") is not None:
                content_block_text += delta.get("thinking", "")
            if delta.get("type") == "signature_delta" and delta.get("signature") is not None:
                content_block_signature = delta.get("signature", "")
        if (content_block := chunk.meta.get("content_block")) is not None and content_block.get(
            "type"
        ) == "redacted_thinking":
            content_block_redacted_thinking = content_block.get("data", "")

    # Finalize the last group
    if current_index is not None:
        if content_block_text:
            formatted_reasoning_contents.append(
                {
                    "reasoning_content": {
                        "reasoning_text": {"text": content_block_text, "signature": content_block_signature},
                    }
                }
            )
        if content_block_redacted_thinking:
            formatted_reasoning_contents.append(
                {"reasoning_content": {"redacted_thinking": content_block_redacted_thinking}}
            )

    # Combine all reasoning texts into a single string for the main reasoning_text field
    final_reasoning_text = ""
    for content in formatted_reasoning_contents:
        if "reasoning_text" in content["reasoning_content"]:
            # mypy somehow thinks that content["reasoning_content"]["reasoning_text"]["text"] can be of type None
            final_reasoning_text += content["reasoning_content"]["reasoning_text"]["text"]  # type: ignore[operator]
        elif "redacted_thinking" in content["reasoning_content"]:
            final_reasoning_text += "[REDACTED]"

    return (
        ReasoningContent(
            reasoning_text=final_reasoning_text, extra={"reasoning_contents": formatted_reasoning_contents}
        )
        if formatted_reasoning_contents
        else None
    )
