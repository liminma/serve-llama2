#
# For API specs, refer to: https://platform.openai.com/docs/api-reference/chat/create
# and https://platform.openai.com/docs/api-reference/completions/create
#
import time
from uuid import uuid4
from enum import Enum

from typing import Literal
from pydantic import BaseModel, Field


class RoleEnum(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'
    FUNCTION = 'function'


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ChatMessage(BaseModel):
    role: RoleEnum
    content: str | None
    name: str = None
    function_call: FunctionCall = None


class Function(BaseModel):
    name: str
    description: str = None
    parameters: dict


class FunctionCallChoiceEnum(str, Enum):
    NONE = 'none'  # default if no functions specified
    AUTP = 'auto'  # default if functions specified


class ChatCompletionRequestBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    functions: list[Function] = None
    function_call: FunctionCallChoiceEnum | dict[str, str] = None
    temperature: float | None = 0.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] = None
    max_tokens: int | None = 4096
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[int, int] = None
    user: str = None


class ChatResponseMessage(BaseModel):
    role: str = 'assistant'
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: Literal['stop', 'length',
                           'function_call', 'content_filter'] = None


class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0  # = completion_tokens + prompt_tokens


class ChatCompletionResponse(BaseModel):
    # refer to https://platform.openai.com/docs/guides/gpt/chat-completions-api
    choices: list[ChatCompletionChoice]
    created: int = Field(default_factory=lambda: int(time.time()))
    id: str = Field(default_factory=lambda: f'chatcmpl-{uuid4().hex}')
    model: str
    object: str = "chat.completion"
    usage: Usage


class CompletionRequestBody(BaseModel):
    # TODO
    pass


class CompletionResponse(BaseModel):
    # TODO
    pass
