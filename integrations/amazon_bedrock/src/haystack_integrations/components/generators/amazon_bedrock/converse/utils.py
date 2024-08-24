from dataclasses import asdict, dataclass, field
import inspect
import json
from typing import Any, Callable, List, Dict, Sequence, Tuple, Union, Optional
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
    format: str
    name: str
    source: DocumentSource


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


def get_stream_message(
    stream: EventStream,
    streaming_callback: Callable[[ConverseStreamingChunk], None],
) -> Tuple[ConverseMessage, Dict[str, Any]]:

    current_block: Union[str, ToolUseBlock] = ""
    current_tool_use_input_str: str = ""
    latest_metadata: Dict[str, Any] = {}
    event_type: str
    current_index: int = 0  # Start with 0 as the first content block seems to be always text 
    # which never starts with a content block start for some reason...

    streamed_contents: List[Union[str, ToolUseBlock]] = []

    for event in stream:
        if "contentBlockStart" in event:
            event_type = "contentBlockStart"
            new_index = event["contentBlockStart"].get("contentBlockIndex", current_index + 1)

            # If index changed, we're starting a new block
            if new_index != current_index:
                if current_block:
                    streamed_contents.append(current_block)
                current_index = new_index
                current_block = ""

            start_of_tool_use = event["contentBlockStart"].get("start")
            if start_of_tool_use:
                current_block = ToolUseBlock(
                    toolUseId=start_of_tool_use["toolUse"]["toolUseId"],
                    name=start_of_tool_use["toolUse"]["name"],
                    input={},
                )

        if "contentBlockDelta" in event:
            event_type = "contentBlockDelta"
            delta = event["contentBlockDelta"].get("delta", {})

            if "text" in delta:
                if isinstance(current_block, str):
                    current_block += delta["text"]
                else:
                    # If we get text when we expected a tool use, start a new string block
                    streamed_contents.append(current_block)
                    current_block = delta["text"]
                    current_index += 1

            if "toolUse" in delta:
                if isinstance(current_block, ToolUseBlock):
                    tool_use_input_delta = delta["toolUse"].get("input")
                    current_tool_use_input_str += tool_use_input_delta
                else:
                    # If we get a tool use when we expected text, start a new ToolUseBlock
                    streamed_contents.append(current_block)
                    current_block = ToolUseBlock(
                        toolUseId=delta["toolUse"]["toolUseId"],
                        name=delta["toolUse"]["name"],
                        input=(json.loads(current_tool_use_input_str)),
                    )
                    current_index += 1

        if "contentBlockStop" in event:
            event_type = "contentBlockStop"
            if isinstance(current_block, ToolUseBlock):
                current_block.input = json.loads(current_tool_use_input_str)
                current_tool_use_input_str = ""
            streamed_contents.append(current_block)
            current_block = ""
            current_index += 1

        if "messageStop" in event:
            event_type = "messageStop"
            latest_metadata["stopReason"] = event["messageStop"].get("stopReason")
            
        if "metadata" in event:
            event_type = "metadata"
        
        if "messageStart" in event:
            event_type = "messageStart"

        latest_metadata.update(event.get("metadata", {}))

        streaming_chunk = ConverseStreamingChunk(
            content=current_block,
            metadata=latest_metadata,
            index=current_index,
            type=event_type,
        )
        streaming_callback(streaming_chunk)

    # Add any remaining content
    if current_block:
        streamed_contents.append(current_block)

    return (
        ConverseMessage(
            role=ConverseRole.ASSISTANT,
            content=ContentBlock.from_assistant(streamed_contents),
        ),
        latest_metadata,
    )
