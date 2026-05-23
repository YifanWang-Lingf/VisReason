from typing import Dict, List, Literal, Union

from openai.types import CompletionUsage
from pydantic import BaseModel, Field

from .utils import convert_image_content_to_openai


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    image: str


ChatContent = List[Union[TextContent, ImageContent]]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, ChatContent]


class ModelInput(BaseModel):
    idx: int
    messages: List[Message] = Field(default_factory=list)

    def get_messages(self) -> List[Dict]:
        return self.model_dump().get("messages")

    def get_openai_messages(self) -> List[Dict]:
        messages = self.model_dump().get("messages")
        for msg in messages:
            if isinstance(msg["content"], str):
                continue
            for content in msg["content"]:
                if content["type"] == "image":
                    convert_image_content_to_openai(content)
        return messages


class ModelResponse(BaseModel):
    idx: int = Field(exclude=True)
    response: Union[str, ChatContent]
    finish_reason: str | None = None
    usage: CompletionUsage | None = None
