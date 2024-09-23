# Copyright (c) Microsoft. All rights reserved.

import json
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from anthropic import AsyncAnthropic
from anthropic.lib.streaming._types import TextEvent
from anthropic.types import (
    ContentBlockStopEvent,
    Message,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    ToolUseBlock,
)
from pydantic import ValidationError

from semantic_kernel.connectors.ai.anthropic.prompt_execution_settings.anthropic_prompt_execution_settings import (
    AnthropicChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.anthropic.settings.anthropic_settings import AnthropicSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_call_choice_configuration import FunctionCallChoiceConfiguration
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceType
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ITEM_TYPES, ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.utils.finish_reason import FinishReason as SemanticKernelFinishReason
from semantic_kernel.exceptions.service_exceptions import (
    ServiceInitializationError,
    ServiceInvalidResponseError,
    ServiceResponseException,
)
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.utils.experimental_decorator import experimental_class
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    trace_chat_completion,
    trace_streaming_chat_completion,
)

# map finish reasons from Anthropic to Semantic Kernel
ANTHROPIC_TO_SEMANTIC_KERNEL_FINISH_REASON_MAP = {
    "end_turn": SemanticKernelFinishReason.STOP,
    "max_tokens": SemanticKernelFinishReason.LENGTH,
    "tool_use": SemanticKernelFinishReason.TOOL_CALLS,
}

logger: logging.Logger = logging.getLogger(__name__)


@experimental_class
class AnthropicChatCompletion(ChatCompletionClientBase):
    """Antropic ChatCompletion class."""

    MODEL_PROVIDER_NAME: ClassVar[str] = "anthropic"
    SUPPORTS_FUNCTION_CALLING: ClassVar[bool] = True

    async_client: AsyncAnthropic

    def __init__(
        self,
        ai_model_id: str | None = None,
        service_id: str | None = None,
        api_key: str | None = None,
        async_client: AsyncAnthropic | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize an AnthropicChatCompletion service.

        Args:
            ai_model_id: Anthropic model name, see
                https://docs.anthropic.com/en/docs/about-claude/models#model-names
            service_id: Service ID tied to the execution settings.
            api_key: The optional API key to use. If provided will override,
                the env vars or .env file value.
            async_client: An existing client to use.
            env_file_path: Use the environment settings file as a fallback
                to environment variables.
            env_file_encoding: The encoding of the environment settings file.
        """
        try:
            anthropic_settings = AnthropicSettings.create(
                api_key=api_key,
                chat_model_id=ai_model_id,
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
            )
        except ValidationError as ex:
            raise ServiceInitializationError("Failed to create Anthropic settings.", ex) from ex

        if not anthropic_settings.chat_model_id:
            raise ServiceInitializationError("The Anthropic chat model ID is required.")

        if not async_client:
            async_client = AsyncAnthropic(
                api_key=anthropic_settings.api_key.get_secret_value(),
            )

        super().__init__(
            async_client=async_client,
            service_id=service_id or anthropic_settings.chat_model_id,
            ai_model_id=anthropic_settings.chat_model_id,
        )

    # region Overriding base class methods

    # Override from AIServiceClientBase
    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        return AnthropicChatPromptExecutionSettings

    # Override from AIServiceClientBase
    @override
    def service_url(self) -> str | None:
        return str(self.async_client.base_url)

    @override
    @trace_chat_completion(MODEL_PROVIDER_NAME)
    async def _inner_get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
    ) -> list["ChatMessageContent"]:
        if not isinstance(settings, AnthropicChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AnthropicChatPromptExecutionSettings)  # nosec

        settings.ai_model_id = settings.ai_model_id or self.ai_model_id
        settings.messages, parsed_system_message = self._prepare_chat_history_for_request(chat_history)
        if settings.system is None and parsed_system_message is not None:
            settings.system = parsed_system_message

        return await self._send_chat_request(settings)

    @override
    @trace_streaming_chat_completion(MODEL_PROVIDER_NAME)
    async def _inner_get_streaming_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
    ) -> AsyncGenerator[list["StreamingChatMessageContent"], Any]:
        if not isinstance(settings, AnthropicChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AnthropicChatPromptExecutionSettings)  # nosec

        messages, parsed_system_message = self._prepare_chat_history_for_request(chat_history, stream=True)
        settings.ai_model_id = settings.ai_model_id or self.ai_model_id
        settings.messages = messages
        if settings.system is None and parsed_system_message is not None:
            settings.system = parsed_system_message

        response = self._send_chat_stream_request(settings)
        if not isinstance(response, AsyncGenerator):
            raise ServiceInvalidResponseError("Expected an AsyncGenerator response.")

        async for message in response:
            yield message

    def _prepare_chat_history_for_request(
        self,
        chat_history: "ChatHistory",
        role_key: str = "role",
        content_key: str = "content",
        stream: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Prepare the chat history for an Anthropic request.

        Allowing customization of the key names for role/author, and optionally overriding the role.

        Args:
            chat_history: The chat history to prepare.
            role_key: The key name for the role/author.
            content_key: The key name for the content/message.
            stream: Whether the request is for a streaming chat.

        Returns:
            A tuple containing the prepared chat history and the first SYSTEM message content.
        """
        system_message_content = None
        remaining_messages = []
        system_message_found = False
        idx = 0
        while idx < len(chat_history.messages):
            message = chat_history.messages[idx]

            # Skip system messages after the first one is found
            if message.role == AuthorRole.SYSTEM:
                if not system_message_found:
                    system_message_content = message.content
                    system_message_found = True

                idx += 1
            elif message.finish_reason == SemanticKernelFinishReason.TOOL_CALLS:
                if not stream:
                    remaining_messages.append({
                        role_key: AuthorRole.ASSISTANT,
                        content_key: message.inner_content.content,
                    })
                else:
                    content = [
                        TextBlock(
                            text=message.items[0].text,
                            type="text"
                        )
                    ]

                    # for remaining items, add them to the content
                    for item in message.items[1:]:
                        tool_use = ToolUseBlock(
                            id=item.id,
                            input=json.loads(item.arguments),
                            name=item.name,
                            type="tool_use"
                        )

                        content.append(tool_use)

                    remaining_messages.append({
                        role_key: AuthorRole.ASSISTANT,
                        content_key: content,
                    })

                idx += 1
                tool_results_message = {
                    role_key: AuthorRole.USER,
                    content_key: [],
                }

                # Add the tool results
                while idx < len(chat_history.messages) and chat_history.messages[idx].role == AuthorRole.TOOL:
                    tool_result = chat_history.messages[idx]
                    tool_results_message["content"].append({
                        "type": "tool_result",
                        "tool_use_id": tool_result.items[0].id,
                        content_key: str(tool_result.items[0].result),
                    })

                    idx += 1

                remaining_messages.append(tool_results_message)
            else:
                # The API requires only role and content keys for the remaining messages
                remaining_messages.append({
                    role_key: getattr(message, role_key),
                    content_key: getattr(message, content_key),
                })

                idx += 1

        return remaining_messages, system_message_content

    # endregion

    def _create_chat_message_content(
        self,
        response: Message,
        content_block: TextBlock | ToolUseBlock,
        response_metadata: dict[str, Any]
    ) -> "ChatMessageContent":
        """Create a chat message content object."""
        items: list[ITEM_TYPES] = self._get_tool_calls_from_message(response)

        if isinstance(content_block, TextBlock):
            items.append(TextContent(text=content_block.text))

        finish_reason = None
        if response.stop_reason:
            finish_reason = ANTHROPIC_TO_SEMANTIC_KERNEL_FINISH_REASON_MAP[response.stop_reason]

        return ChatMessageContent(
            inner_content=response,
            ai_model_id=self.ai_model_id,
            metadata=response_metadata,
            role=AuthorRole.ASSISTANT,
            items=items,
            finish_reason=finish_reason,
        )

    def _create_streaming_chat_message_content(
        self,
        stream_event: TextEvent | ContentBlockStopEvent | RawMessageDeltaEvent,
        metadata: dict[str, Any] = {},
    ) -> StreamingChatMessageContent:
        """Create a streaming chat message content object from a content block."""
        items: list[FunctionCallContent | StreamingTextContent] = []
        finish_reason = None

        if isinstance(stream_event, TextEvent):
            items.append(StreamingTextContent(choice_index=0, text=stream_event.text))
        elif isinstance(stream_event, ContentBlockStopEvent):
            tool_use_block = stream_event.content_block
            items.append(
                FunctionCallContent(
                    id=tool_use_block.id,
                    index=stream_event.index,
                    name=tool_use_block.name,
                    arguments=json.dumps(tool_use_block.input) if tool_use_block.input else None,
                )
            )
        elif isinstance(stream_event, RawMessageDeltaEvent):
            finish_reason = ANTHROPIC_TO_SEMANTIC_KERNEL_FINISH_REASON_MAP[stream_event.delta.stop_reason]
            metadata["usage"]["output_tokens"] = stream_event.usage.output_tokens

        return StreamingChatMessageContent(
            choice_index=0,
            inner_content=stream_event,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole.ASSISTANT,
            finish_reason=finish_reason,
            items=items,
        )

    def update_settings_from_function_call_configuration_anthropic(
        self,
        function_choice_configuration: FunctionCallChoiceConfiguration,
        settings: "PromptExecutionSettings",
        type: "FunctionChoiceType",
    ) -> None:
        """Update the settings from a FunctionChoiceConfiguration."""
        if (
            function_choice_configuration.available_functions
            and hasattr(settings, "tools")
        ):
            settings.tools = [
                self.kernel_function_metadata_to_function_call_format_anthropic(f)
                for f in function_choice_configuration.available_functions
            ]
            
            if (
                (settings.function_choice_behavior
                and settings.function_choice_behavior.type_ == FunctionChoiceType.REQUIRED)
                or type == FunctionChoiceType.REQUIRED
            ):
                settings.tool_choice = {"type": "any"}
            elif type == FunctionChoiceType.AUTO:
                settings.tool_choice = {"type": type.value}
            else:
                raise ValueError(
                    f"AnthropicChatCompletion does not support the {type} function choice type."
                )

    def kernel_function_metadata_to_function_call_format_anthropic(
        self,
        metadata: KernelFunctionMetadata,
    ) -> dict[str, Any]:
        """Convert the kernel function metadata to function calling format."""
        return {
            "name": metadata.fully_qualified_name,
            "description": metadata.description or "",
            "input_schema": {
                "type": "object",
                "properties": {p.name: p.schema_data for p in metadata.parameters},
                "required": [p.name for p in metadata.parameters if p.is_required],
            },
        }

    @override
    def _update_function_choice_settings_callback(
        self,
    ):
        return self.update_settings_from_function_call_configuration_anthropic

    async def _send_chat_request(self, settings: AnthropicChatPromptExecutionSettings) -> list["ChatMessageContent"]:
        """Send the chat request."""
        try:
            kwargs = settings.model_dump(
                exclude={
                    "service_id",
                    "extension_data",
                    "messages"
                },
                exclude_none=True,
                by_alias=True,
            )

            response = await self.async_client.messages.create(messages=settings.messages, **kwargs)
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the request",
                ex,
            ) from ex

        response_metadata: dict[str, Any] = {"id": response.id}
        if hasattr(response, "usage") and response.usage is not None:
            response_metadata["usage"] = response.usage

        return [
            self._create_chat_message_content(response, content_block, response_metadata)
            for content_block in response.content
        ]

    async def _send_chat_stream_request(
        self, settings: AnthropicChatPromptExecutionSettings
    ) -> AsyncGenerator[list["StreamingChatMessageContent"], None]:
        """Send the chat stream request."""
        try:
            kwargs = settings.model_dump(
                exclude={
                    "service_id",
                    "extension_data",
                },
                exclude_none=True,
                by_alias=True,
            )

            async with self.async_client.messages.stream(**kwargs) as stream:
                metadata: dict[str, Any] = {"usage": {}, "id": None}

                async for stream_event in stream:
                    if isinstance(stream_event, RawMessageStartEvent):
                        metadata["usage"]["input_tokens"] = stream_event.message.usage.input_tokens
                        metadata["id"] = stream_event.message.id
                    elif isinstance(stream_event, (TextEvent, RawMessageDeltaEvent)) or \
                        (isinstance(stream_event, ContentBlockStopEvent) and
                         stream_event.content_block.type == "tool_use"):
                        yield [self._create_streaming_chat_message_content(stream_event, metadata)]
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the request",
                ex,
            ) from ex

    def _get_tool_calls_from_message(self, message) -> list[FunctionCallContent]:
        """Get tool calls from a content blocks."""
        tool_calls = []

        for idx, content_block in enumerate(message.content):
            if isinstance(content_block, ToolUseBlock):
                tool_calls.append(
                    FunctionCallContent(
                        id=content_block.id,
                        index=idx,
                        name=content_block.name,
                        arguments=content_block.input,
                    )
                )

        return tool_calls

    @override
    def _reset_function_choice_settings(self, settings: "PromptExecutionSettings") -> None:
        if hasattr(settings, "tool_choice"):
            settings.tool_choice = None
        if hasattr(settings, "tools"):
            settings.tools = None
