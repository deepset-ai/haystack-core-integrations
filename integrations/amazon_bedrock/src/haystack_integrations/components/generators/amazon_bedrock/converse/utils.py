from dataclasses import asdict, dataclass, field
import inspect
import json
from typing import Any, Callable, List, Dict, Tuple, Union, Optional
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
    input: Dict


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


def get_stream_message(
    stream: EventStream,
    streaming_callback: Callable[[ConverseStreamingChunk], None],
) -> Tuple[ConverseMessage, Dict[str, Any]]:
    streaming_chunks: List[ConverseStreamingChunk] = []
    tool_use_dict = {}
    str_message = ""
    latest_metadata = {}
    content_index = 0  # used to keep track of the current str/tool use alternance
    current_tool_use_str = ""
    for event in stream:
        if "contentBlockStart" in event:
            if len(current_tool_use_str) > 0 and content_index != event["contentBlockStart"].get("contentBlockIndex"):
                tool_use_dict["input"] = current_tool_use_str
                current_tool_use_str = ""
                
            start = event["contentBlockStart"].get("start")
            content_index = event["contentBlockStart"].get("contentBlockIndex", content_index)

            if start:
                tool_use_dict["toolUseId"] = start["toolUse"]["toolUseId"]
                tool_use_dict["name"] = start["toolUse"]["name"]

        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta")

            if "text" in delta:
                str_message += delta["text"]
            if "toolUse" in delta:
                current_tool_use_str += delta["toolUse"]["input"]

        if "contentBlockStop" in event:
            content_index += 1 # start a new str/tool use alternation

        if "messageStop" in event:
            stop_reason = event["messageStop"].get("stopReason")
            latest_metadata["stopReason"] = stop_reason

        latest_metadata.update(event.get("metadata", {}))

        block_content = ToolUseBlock(**tool_use_dict) if len(tool_use_dict) == 3 else str_message

        streaming_chunk = ConverseStreamingChunk(
            content=block_content,
            metadata=event.get("metadata", {}),
            index=content_index,
        )

        streaming_callback(streaming_chunk)
        streaming_chunks.append(streaming_chunk)
    return (
        ConverseMessage(
            role=ConverseRole.ASSISTANT,
            content=ContentBlock([block_content]),
        ),
        latest_metadata,
    )
