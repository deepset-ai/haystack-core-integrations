from dataclasses import dataclass
from enum import Enum
from typing import List, Union, Optional


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


from dataclasses import dataclass, asdict
from typing import Union, Optional


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
            else:
                raise NotImplementedError
        return res

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


class ConverseRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConverseMessage:
    def __init__(self, role: ConverseRole, content: ContentBlock):
        self.role = role
        self.content = content

    @staticmethod
    def from_user(
        content: List[
            Union[DocumentBlock, GuardrailConverseContentBlock, ImageBlock, str, ToolResultBlock, ToolUseBlock]
        ]
    ) -> "ConverseMessage":
        return ConverseMessage(
            ConverseRole.USER,
            ContentBlock(
                content=content,
            ),
        )

    def to_dict(self):
        return {
            "role": self.role.value,
            "content": self.content.to_dict(),
        }
