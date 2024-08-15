from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class DocumentBlock:
    # Placeholder for DocumentBlock attributes
    pass


@dataclass
class GuardrailConverseContentBlock:
    # Placeholder for GuardrailConverseContentBlock attributes
    pass


@dataclass
class ImageBlock:
    # Placeholder for ImageBlock attributes
    pass


@dataclass
class ToolResultBlock:
    # Placeholder for ToolResultBlock attributes
    pass


@dataclass
class ToolUseBlock:
    # Placeholder for ToolUseBlock attributes
    pass


@dataclass
class ContentBlock:
    content: Union[DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, ToolUseBlock]

    def __post_init__(self):
        if not isinstance(
            self.content, (DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, ToolUseBlock)
        ):
            raise ValueError(
                "Invalid content type. Must be one of DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, or ToolUseBlock"
            )

    @property
    def document(self) -> Optional[DocumentBlock]:
        return self.content if isinstance(self.content, DocumentBlock) else None

    @property
    def guard_content(self) -> Optional[GuardrailConverseContentBlock]:
        return self.content if isinstance(self.content, GuardrailConverseContentBlock) else None

    @property
    def image(self) -> Optional[ImageBlock]:
        return self.content if isinstance(self.content, ImageBlock) else None

    @property
    def text(self) -> Optional[str]:
        return self.content if isinstance(self.content, str) else None

    @property
    def tool_result(self) -> Optional[ToolResultBlock]:
        return self.content if isinstance(self.content, ToolResultBlock) else None

    @property
    def tool_use(self) -> Optional[ToolUseBlock]:
        return self.content if isinstance(self.content, ToolUseBlock) else None
