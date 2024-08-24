from dataclasses import asdict, dataclass, field
import inspect
import json
import logging
from typing import Any, Callable, List, Dict, Literal, Sequence, Tuple, Union, Optional
from enum import Enum
from botocore.eventstream import EventStream


@dataclass
class ToolSpec:
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class Tool:
    toolSpec: ToolSpec


@dataclass
class ToolChoice:
    auto: Dict = field(default_factory=dict)
    any: Dict = field(default_factory=dict)
    tool: Optional[Dict[str, str]] = None


@dataclass
class ToolConfig:
    tools: List[Tool]
    toolChoice: Optional[ToolChoice] = None

    def __post_init__(self):
        if self.toolChoice and sum(bool(v) for v in vars(self.toolChoice).values()) != 1:
            raise ValueError("Only one of 'auto', 'any', or 'tool' can be set in toolChoice")

        if self.toolChoice and self.toolChoice.tool:
            if 'name' not in self.toolChoice.tool:
                raise ValueError("'name' is required when 'tool' is specified in toolChoice")

    @staticmethod
    def from_functions(functions: List[Callable]) -> 'ToolConfig':
        tools = []
        for func in functions:
            tool_spec = ToolSpec(
                name=func.__name__,
                description=func.__doc__,
                inputSchema={
                    "json": {
                        "type": "object",
                        "properties": {param: {"type": "string"} for param in inspect.signature(func).parameters},
                        "required": list(inspect.signature(func).parameters.keys()),
                    }
                },
            )
            tools.append(Tool(toolSpec=tool_spec))

        return ToolConfig(tools=tools)

    @classmethod
    def from_dict(cls, config: Dict) -> 'ToolConfig':
        tools = [Tool(ToolSpec(**tool['toolSpec'])) for tool in config.get('tools', [])]

        tool_choice = None
        if 'toolChoice' in config:
            tc = config['toolChoice']
            if 'auto' in tc:
                tool_choice = ToolChoice(auto=tc['auto'])
            elif 'any' in tc:
                tool_choice = ToolChoice(any=tc['any'])
            elif 'tool' in tc:
                tool_choice = ToolChoice(tool={'name': tc['tool']['name']})

        return cls(tools=tools, toolChoice=tool_choice)

    def to_dict(self) -> Dict[str, Any]:
        result = {"tools": [{"toolSpec": asdict(tool.toolSpec)} for tool in self.tools]}
        if self.toolChoice:
            tool_choice: Dict[str, Dict[str, Any]] = {}
            if self.toolChoice.auto:
                tool_choice["auto"] = self.toolChoice.auto
            elif self.toolChoice.any:
                tool_choice["any"] = self.toolChoice.any
            elif self.toolChoice.tool:
                tool_choice["tool"] = self.toolChoice.tool
            result["toolChoice"] = [tool_choice]
        return result


@dataclass
class DocumentSource:
    bytes: bytes


@dataclass
class DocumentBlock:
    SUPPORTED_FORMATS = Literal['pdf', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'txt', 'md']
    format: SUPPORTED_FORMATS
    name: str
    source: bytes


@dataclass
class GuardrailConverseContentBlock:
    text: str
    qualifiers: List[str] = field(default_factory=list)


@dataclass
class ImageSource:
    bytes: bytes


@dataclass
class ImageBlock:
    format: str
    source: ImageSource


@dataclass
class ToolResultContentBlock:
    json: Optional[Dict] = None
    text: Optional[str] = None
    image: Optional[ImageBlock] = None
    document: Optional[DocumentBlock] = None


@dataclass
class ToolResultBlock:
    toolUseId: str
    content: List[ToolResultContentBlock]
    status: Optional[str] = None


@dataclass
class ToolUseBlock:
    toolUseId: str
    name: str
    input: Dict[str, Any]


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ContentBlock:
    content: List[Union[DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, ToolUseBlock]]

    def __post_init__(self):
        if not isinstance(self.content, list):
            raise ValueError("Content must be a list")

        for item in self.content:
            if not isinstance(
                item, (DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, ToolUseBlock)
            ):
                raise ValueError(
                    f"Invalid content type: {type(item)}. Each item must be one of DocumentBlock, "
                    "GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, or ToolUseBlock"
                )

    @staticmethod
    def from_assistant(content: Sequence[Union[str, ToolUseBlock]]) -> 'ContentBlock':
        return ContentBlock(content=list(content))

    def to_dict(self):
        res = []
        for item in self.content:
            if isinstance(item, str):
                res.append({"text": item})
            elif isinstance(item, DocumentBlock):
                res.append({"document": asdict(item)})
            elif isinstance(item, GuardrailConverseContentBlock):
                res.append({"guardContent": asdict(item)})
            elif isinstance(item, ImageBlock):
                res.append({"image": asdict(item)})
            elif isinstance(item, ToolResultBlock):
                res.append({"toolResult": asdict(item)})
            elif isinstance(item, ToolUseBlock):
                res.append({"toolUse": asdict(item)})
            else:
                raise ValueError(f"Unsupported content type: {type(item)}")
        return res


@dataclass
class ConverseMessage:
    role: ConverseRole
    content: ContentBlock

    @staticmethod
    def from_user(
        content: List[
            Union[
                DocumentBlock,
                GuardrailConverseContentBlock,
                ImageBlock,
                str,
                ToolUseBlock,
                ToolResultBlock,
            ],
        ],
    ) -> "ConverseMessage":
        return ConverseMessage(
            ConverseRole.USER,
            ContentBlock(
                content=content,
            ),
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConverseMessage":
        role = ConverseRole(data['role'])
        content_blocks = []

        for item in data['content']:
            if 'text' in item:
                content_blocks.append(item['text'])
            elif 'image' in item:
                content_blocks.append(ImageBlock(**item['image']))
            elif 'document' in item:
                content_blocks.append(DocumentBlock(**item['document']))
            elif 'toolUse' in item:
                content_blocks.append(ToolUseBlock(**item['toolUse']))
            elif 'toolResult' in item:
                content_blocks.append(ToolResultBlock(**item['toolResult']))
            elif 'guardContent' in item:
                content_blocks.append(GuardrailConverseContentBlock(**item['guardContent']))
            else:
                raise ValueError(f"Unknown content type in message: {item}")

        return ConverseMessage(role, ContentBlock(content=content_blocks))

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content.to_dict(),
        }


@dataclass
class ConverseStreamingChunk:
    content: Union[str, ToolUseBlock]
    metadata: Dict[str, Any]
    index: int = 0
    type: str = ""


@dataclass
class StreamEvent:
    type: str
    data: Dict[str, Any]


def parse_event(event: Dict[str, Any]) -> StreamEvent:
    for key in ['contentBlockStart', 'contentBlockDelta', 'contentBlockStop', 'messageStop', 'messageStart']:
        if key in event:
            return StreamEvent(type=key, data=event[key])
    return StreamEvent(type='metadata', data=event.get('metadata', {}))


def handle_content_block_start(event: StreamEvent, current_index: int) -> Tuple[int, Union[str, ToolUseBlock]]:
    new_index = event.data.get('contentBlockIndex', current_index + 1)
    start_of_tool_use = event.data.get('start')
    if start_of_tool_use:
        return new_index, ToolUseBlock(
            toolUseId=start_of_tool_use['toolUse']['toolUseId'],
            name=start_of_tool_use['toolUse']['name'],
            input={},
        )
    return new_index, ""


def handle_content_block_delta(
    event: StreamEvent, current_block: Union[str, ToolUseBlock], current_tool_use_input_str: str
) -> Tuple[Union[str, ToolUseBlock], str]:
    delta = event.data.get('delta', {})
    if 'text' in delta:
        if isinstance(current_block, str):
            return current_block + delta['text'], current_tool_use_input_str
        else:
            return delta['text'], current_tool_use_input_str
    if 'toolUse' in delta:
        if isinstance(current_block, ToolUseBlock):
            return current_block, current_tool_use_input_str + delta['toolUse'].get('input', '')
        else:
            return ToolUseBlock(
                toolUseId=delta['toolUse']['toolUseId'],
                name=delta['toolUse']['name'],
                input={},
            ), delta['toolUse'].get('input', '')
    return current_block, current_tool_use_input_str


def get_stream_message(
    stream: EventStream,
    streaming_callback: Callable[[ConverseStreamingChunk], None],
) -> Tuple[ConverseMessage, Dict[str, Any]]:
    current_block: Union[str, ToolUseBlock] = ""
    current_tool_use_input_str: str = ""
    latest_metadata: Dict[str, Any] = {}
    current_index: int = 0
    streamed_contents: List[Union[str, ToolUseBlock]] = []

    try:
        for raw_event in stream:
            event = parse_event(raw_event)

            if event.type == 'contentBlockStart':
                if current_block:
                    if isinstance(current_block, str) and streamed_contents and isinstance(streamed_contents[-1], str):
                        streamed_contents[-1] += current_block
                    else:
                        streamed_contents.append(current_block)
                current_index, current_block = handle_content_block_start(event, current_index)

            elif event.type == 'contentBlockDelta':
                new_block, new_input_str = handle_content_block_delta(event, current_block, current_tool_use_input_str)
                if isinstance(new_block, ToolUseBlock) and new_block != current_block:
                    if current_block:
                        if (
                            isinstance(current_block, str)
                            and streamed_contents
                            and isinstance(streamed_contents[-1], str)
                        ):
                            streamed_contents[-1] += current_block
                        else:
                            streamed_contents.append(current_block)
                    current_index += 1
                current_block, current_tool_use_input_str = new_block, new_input_str

            elif event.type == 'contentBlockStop':
                if isinstance(current_block, ToolUseBlock):
                    current_block.input = json.loads(current_tool_use_input_str)
                    current_tool_use_input_str = ""
                    streamed_contents.append(current_block)
                elif isinstance(current_block, str):
                    if streamed_contents and isinstance(streamed_contents[-1], str):
                        streamed_contents[-1] += current_block
                    else:
                        streamed_contents.append(current_block)
                current_block = ""
                current_index += 1

            elif event.type == 'messageStop':
                latest_metadata["stopReason"] = event.data.get("stopReason")

            latest_metadata.update(event.data if event.type == 'metadata' else {})

            streaming_chunk = ConverseStreamingChunk(
                content=current_block,
                metadata=latest_metadata,
                index=current_index,
                type=event.type,
            )
            streaming_callback(streaming_chunk)

    except Exception as e:
        logging.error(f"Error processing stream: {str(e)}")
        raise

    # Add any remaining content
    if current_block:
        if isinstance(current_block, str) and streamed_contents and isinstance(streamed_contents[-1], str):
            streamed_contents[-1] += current_block
        else:
            streamed_contents.append(current_block)

    return (
        ConverseMessage(
            role=ConverseRole.ASSISTANT,
            content=ContentBlock.from_assistant(streamed_contents),
        ),
        latest_metadata,
    )
