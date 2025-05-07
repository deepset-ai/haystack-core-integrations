from haystack.dataclasses import StreamingChunk


def print_streaming_chunk(chunk: StreamingChunk):
    """
    Callback function to handle and display streaming output chunks.

    This function processes a `StreamingChunk` object by:
    - Printing tool call metadata (if any), including function names and arguments, as they arrive.
    - Printing tool call results when available.
    - Printing the main content (e.g., text tokens) of the chunk as it is received.

    The function outputs data directly to stdout and flushes output buffers to ensure immediate display during
    streaming.

    :param chunk: A chunk of streaming data containing content and optional metadata, such as tool calls and
        tool results.
    """
    if tool_calls := chunk.meta.get("tool_calls", {}):
        if tool_calls.get("name"):

            # print the tool name
            print("\n[TOOL CALL]\n", flush=True, end="")
            print(f"Tool: {tool_calls['name']}", flush=True, end="")

        # print the tool arguments
        if arguments := tool_calls.get("arguments"):
            if arguments.startswith("{"):
                print("\nArguments: ", flush=True, end="")
            print(arguments, flush=True, end="")
            if arguments.endswith("}"):
                print("\n\n", flush=True, end="")

    # Print tool call results if available (from ToolInvoker)
    if chunk.meta.get("tool_result"):
        print(f"[TOOL RESULT]\n{chunk.meta['tool_result']}\n\n", flush=True, end="")

    # Print the main content of the chunk
    if chunk.content:
        print(chunk.content, flush=True, end="")
