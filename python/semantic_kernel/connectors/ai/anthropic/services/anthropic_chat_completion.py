# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any, ClassVar
import json
from functools import reduce

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    Message,
    RawContentBlockDeltaEvent,
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
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ITEM_TYPES, ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.connectors.ai.function_calling_utils import (
    merge_function_results,
    # update_settings_from_function_call_configuration,
)
from semantic_kernel.contents.streaming_chat_message_content import ITEM_TYPES as STREAMING_ITEM_TYPES
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.utils.finish_reason import FinishReason as SemanticKernelFinishReason
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError, ServiceResponseException, ServiceInvalidExecutionSettingsError
from semantic_kernel.utils.experimental_decorator import experimental_class
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import trace_chat_completion

# map finish reasons from Anthropic to Semantic Kernel
ANTHROPIC_TO_SEMANTIC_KERNEL_FINISH_REASON_MAP = {
    "end_turn": SemanticKernelFinishReason.STOP,
    "max_tokens": SemanticKernelFinishReason.LENGTH,
    "tool_use": SemanticKernelFinishReason.TOOL_CALLS,
}

def update_settings_from_function_call_configuration(
    function_choice_configuration: "FunctionCallChoiceConfiguration",
    settings: "PromptExecutionSettings",
    type: str,
) -> None:
    """Update the settings from a FunctionChoiceConfiguration."""
    if (
        function_choice_configuration.available_functions
        and hasattr(settings, "tools")
    ):
        settings.tools = [
            kernel_function_metadata_to_function_call_format(f)
            for f in function_choice_configuration.available_functions
        ]

def kernel_function_metadata_to_function_call_format(
    metadata: "KernelFunctionMetadata",
) -> dict[str, Any]:
    """Convert the kernel function metadata to function calling format."""

    return {
        "name": metadata.fully_qualified_name,
        "description": metadata.description or "",
        "input_schema": {
            "type": "object",
            "properties": { p.name: p.schema_data for p in metadata.parameters },
            "required": [p.name for p in metadata.parameters if p.is_required],
        },
    }

logger: logging.Logger = logging.getLogger(__name__)


@experimental_class
class AnthropicChatCompletion(ChatCompletionClientBase):
    """Antropic ChatCompletion class."""

    MODEL_PROVIDER_NAME: ClassVar[str] = "anthropic"
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

    @override
    @trace_chat_completion(MODEL_PROVIDER_NAME)
    async def get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> list["ChatMessageContent"]:
        """Executes a chat completion request and returns the result.

        Args:
            chat_history: The chat history to use for the chat completion.
            settings: The settings to use for the chat completion request.
            kwargs: The optional arguments.

        Returns:
            The completion result(s).
        """
        if not isinstance(settings, AnthropicChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AnthropicChatPromptExecutionSettings)  # nosec

        if not settings.ai_model_id:
            settings.ai_model_id = self.ai_model_id

        settings.messages = self._prepare_chat_history_for_request(chat_history)
    
        kernel = kwargs.get("kernel", None)
        if settings.function_choice_behavior is not None:
            if kernel is None:
                raise ServiceInvalidExecutionSettingsError("The kernel is required for Anthropic tool calls.")

        self._prepare_settings(settings, chat_history, stream_request=False, kernel=kernel)
        if settings.function_choice_behavior is None or (
            settings.function_choice_behavior and not settings.function_choice_behavior.auto_invoke_kernel_functions
        ):
            return await self._send_chat_request(settings)

        # loop for auto-invoke function calls
        for request_index in range(settings.function_choice_behavior.maximum_auto_invoke_attempts):
            completions = await self._send_chat_request(settings)
            
            function_calls = [item for item in completions[0].items if isinstance(item, FunctionCallContent)]
            fc_count = len(function_calls)
            if fc_count == 0:
                return completions

            chat_history.add_message(message=completions[0])

            results = await asyncio.gather(
                *[
                    kernel.invoke_function_call(
                        function_call=function_call,
                        chat_history=chat_history,
                        arguments=kwargs.get("arguments", None),
                        function_call_count=fc_count,
                        request_index=request_index,
                        function_behavior=settings.function_choice_behavior,
                    )
                    for function_call in function_calls
                ],
            )

            if any(result.terminate for result in results if result is not None):
                return merge_function_results(chat_history.messages[-len(results) :])
            
            self._update_settings(settings, chat_history, kernel=kernel)
        else:
            settings.function_choice_behavior.auto_invoke_kernel_functions = False
            return await self._send_chat_request(settings)

    @override
    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: PromptExecutionSettings,
        **kwargs: Any,
    ) -> AsyncGenerator[list[StreamingChatMessageContent], Any]:
        """Executes a streaming chat completion request and returns the result.

        Args:
            chat_history: The chat history to use for the chat completion.
            settings: The settings to use for the chat completion request.
            kwargs: The optional arguments.

        Yields:
            A stream of StreamingChatMessageContent.
        """
        if not isinstance(settings, AnthropicChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AnthropicChatPromptExecutionSettings)  # nosec

        if not settings.ai_model_id:
            settings.ai_model_id = self.ai_model_id

        kernel = kwargs.get("kernel", None)
        if settings.function_choice_behavior is not None:
            if kernel is None:
                raise ServiceInvalidExecutionSettingsError("The kernel is required for OpenAI tool calls.")

        self._prepare_settings(settings, chat_history, stream_request=True, kernel=kernel)
        request_attempts = (
            settings.function_choice_behavior.maximum_auto_invoke_attempts
            if (settings.function_choice_behavior and settings.function_choice_behavior.auto_invoke_kernel_functions)
            else 1
        )

        for request_index in range(request_attempts):
            all_messages: list[StreamingChatMessageContent] = []
            function_call_returned = False

            settings.messages = self._prepare_chat_history_for_request(chat_history)
            async for messages in self._send_chat_stream_request(settings):
                for msg in messages:
                    if msg is not None:
                        all_messages.append(msg)
                        if any(isinstance(item, FunctionCallContent) for item in msg.items):
                            function_call_returned = True
                yield messages

            if (
                settings.function_choice_behavior is None
                or (
                    settings.function_choice_behavior
                    and not settings.function_choice_behavior.auto_invoke_kernel_functions
                )
                or not function_call_returned
            ):
                return

            full_completion: StreamingChatMessageContent = reduce(lambda x, y: x + y, all_messages)
            function_calls = [item for item in full_completion.items if isinstance(item, FunctionCallContent)]
            chat_history.add_message(message=full_completion)

            fc_count = len(function_calls)
            results = await asyncio.gather(
                *[
                    self._process_function_call(
                        function_call=function_call,
                        chat_history=chat_history,
                        kernel=kernel,
                        arguments=kwargs.get("arguments", None),
                        function_call_count=fc_count,
                        request_index=request_index,
                        function_call_behavior=settings.function_choice_behavior,
                    )
                    for function_call in function_calls
                ],
            )
            if any(result.terminate for result in results if result is not None):
                yield merge_function_results(chat_history.messages[-len(results) :])  # type: ignore
                break

            self._update_settings(settings, chat_history, kernel=kernel)

    def _create_chat_message_content(
        self, response: Message, content: TextBlock | ToolUseBlock, response_metadata: dict[str, Any]
    ) -> "ChatMessageContent":
        """Create a chat message content object."""
        items: list[ITEM_TYPES] = self._get_tool_calls_from_message(response)

        if isinstance(content, TextBlock):
            items.append(TextContent(text=content.text))

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
        stream_event: RawContentBlockDeltaEvent | RawMessageDeltaEvent | ContentBlockStopEvent,
        content_block_idx: int,
        role: str | None = None,
        metadata: dict[str, Any] = {},
        function_call_dict: dict[str, Any] | None = None,
    ) -> StreamingChatMessageContent:
        """Create a streaming chat message content object from a content block."""
        items: list[Any] = []
        text_content = ""

        if hasattr(stream_event, "delta") and hasattr(stream_event.delta, "text"):
            text_content = stream_event.delta.text
            items.append(StreamingTextContent(choice_index=content_block_idx, text=text_content))
        elif function_call_dict:
            items.append(
                FunctionCallContent(
                    id=function_call_dict["id"],
                    index=content_block_idx,
                    name=function_call_dict["name"],
                    arguments=function_call_dict["arguments"],
                )
            )

        finish_reason = None
        if isinstance(stream_event, RawMessageDeltaEvent):
            if stream_event.delta.stop_reason:
                finish_reason = ANTHROPIC_TO_SEMANTIC_KERNEL_FINISH_REASON_MAP[stream_event.delta.stop_reason]

            metadata["usage"]["output_tokens"] = stream_event.usage.output_tokens

        return StreamingChatMessageContent(
            choice_index=content_block_idx,
            inner_content=stream_event,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole.ASSISTANT,
            finish_reason=finish_reason,
            items=items,
        )
    
    async def _send_chat_request(self, settings: AnthropicChatPromptExecutionSettings) -> list["ChatMessageContent"]:
        """Send the chat request."""

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
        assert isinstance(response, Message)  # nosec

        response_metadata: dict[str, Any] = {"id": response.id }
        if hasattr(response, "usage") and response.usage is not None:
            response_metadata["usage"] = response.usage

        return [self._create_chat_message_content(response, content_block, response_metadata) for content_block in response.content]

    async def _send_chat_stream_request(
        self, settings: AnthropicChatPromptExecutionSettings
    ) -> AsyncGenerator[list["StreamingChatMessageContent"], None]:
        """Send the chat stream request."""

        try:
            async with self.async_client.messages.stream(**settings.prepare_settings_dict()) as stream:
                author_role = None
                metadata: dict[str, Any] = {"usage": {}, "id": None}
                content_block_idx: int = 0
                function_call_dict: dict[str, Any] | None = None

                async for stream_event in stream:
                    if isinstance(stream_event, RawMessageStartEvent):
                        author_role = stream_event.message.role
                        metadata["usage"]["input_tokens"] = stream_event.message.usage.input_tokens
                        metadata["id"] = stream_event.message.id
                    elif isinstance(stream_event, ContentBlockStartEvent) and stream_event.type == "tool_use":
                        function_call_dict = {
                            "id": stream_event.id,
                            "name": stream_event.name,
                            "arguments": "",
                        }
                    elif isinstance(stream_event, (RawContentBlockDeltaEvent, RawMessageDeltaEvent)):
                        if stream_event.delta and hasattr(stream_event.delta, "partial_json"):
                            function_call_dict["arguments"] += stream_event.delta.partial_json
                        else:
                            yield [
                                self._create_streaming_chat_message_content(
                                    stream_event, content_block_idx, author_role, metadata
                                )
                            ]
                    elif isinstance(stream_event, ContentBlockStopEvent):
                        if function_call_dict:
                            yield [
                                self._create_streaming_chat_message_content(
                                    stream_event, content_block_idx, author_role, metadata, function_call_dict
                                )
                            ]
                            function_call_dict = None

                        content_block_idx += 1

        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the request",
                ex,
            ) from ex
        
    def _get_tool_calls_from_message(self, message) -> list[FunctionCallContent]:
        """Get tool calls from a chat choice."""
        output = []

        for idx, content_block in enumerate(message.content):
            if isinstance(content_block, ToolUseBlock):
                output.append(
                    FunctionCallContent(
                        id=content_block.id,
                        index=idx,
                        name=content_block.name,
                        arguments=content_block.input,
                    )
                )

        return output
    
    def _update_settings(
        self,
        settings: AnthropicChatPromptExecutionSettings,
        chat_history: ChatHistory,
        kernel: "Kernel | None" = None,
    ) -> None:
        """Update the settings with the chat history."""

        anthropic_messages = []
        for message in chat_history.messages:
            if message.role == AuthorRole.USER:
                anthropic_messages.append({"role": "user", "content": message.content})
            elif message.role == AuthorRole.ASSISTANT:
                if message.finish_reason == SemanticKernelFinishReason.TOOL_CALLS:
                    # add assistent message
                    anthropic_messages.append({"role": "assistant", "content": message.inner_content.content})
                else:
                    anthropic_messages.append({"role": "assistant", "content": message.content})
            elif message.role == AuthorRole.TOOL:
                # add tool result to previous user message or create a new user message
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    anthropic_messages[-1]["content"].append({"type": "tool_result", "tool_use_id": message.items[0].id, "content": str(message.items[0].result)})
                else:
                    anthropic_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.items[0].id, "content": str(message.items[0].result)}]})

        settings.messages = anthropic_messages

        if settings.function_choice_behavior and kernel:
            settings.function_choice_behavior.configure(
                kernel=kernel,
                update_settings_callback=update_settings_from_function_call_configuration,
                settings=settings,
            )

    def _prepare_settings(
        self,
        settings: AnthropicChatPromptExecutionSettings,
        chat_history: ChatHistory,
        stream_request: bool = False,
        kernel: "Kernel | None" = None,
    ) -> None:
        """Prepare the prompt execution settings for the chat request."""
        settings.stream = stream_request
        if not settings.ai_model_id:
            settings.ai_model_id = self.ai_model_id
        self._update_settings(settings=settings, chat_history=chat_history, kernel=kernel)

    def get_prompt_execution_settings_class(self) -> "type[AnthropicChatPromptExecutionSettings]":
        """Create a request settings object."""
        return AnthropicChatPromptExecutionSettings
