import asyncio
import json
from typing import List, Dict, Any, Optional, Union, Awaitable

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from neuralcore.utils.exceptions_handler import ConfirmationRequired
from neuralcore.actions.actions import Action, ActionSet
from neuralcore.actions.manager import DynamicActionManager
from neuralcore.utils.text_tokenizer import TextTokenizer
from neuralcore.utils.formatting import (
    is_valid_json,
    prepare_chat_messages,
    drain_queue_to_string,
)
from typing import AsyncIterator, Callable, Tuple

import numpy as np

from neuralcore.utils.logger import Logger


EMIT_INTERVAL = 0.05  # seconds (throttle)
MIN_CHARS_DELTA = 3  # don't spam tiny updates


ToolProvider = Union[ActionSet, DynamicActionManager, List[Dict[str, Any]]]
ToolExecutorGetter = Optional[Callable[[str], Optional["Action"]]]


logger = Logger.get_logger()


async def embed_single_chunk(
    client: AsyncOpenAI,
    chunk: str,
    model: str,
) -> Optional[np.ndarray]:
    try:
        response = await client.embeddings.create(model=model, input=chunk.strip())
        vec = response.data[0].embedding
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding chunk failed: {str(e)[:180]}...")
        return None


class LLMClient:
    _global_cancel_requested = False

    def __init__(
        self,
        base_url: str,
        model: str,
        name: Optional[str] = "main",
        tokenizer: Optional[str] = None,
        api_key: str = "not-needed",
        extra_body: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body_default = extra_body or {}
        self.api_key = api_key
        self.system_prompt = system_prompt or "You are a helpful assistant."

        # --- tokenizer resolution ---
        if isinstance(tokenizer, str):
            # Explicit tokenizer source passed
            self.tokenizer = TextTokenizer(tokenizer_source=tokenizer)
        else:
            # Fall back to singleton initialized via config (client name)
            self.tokenizer = TextTokenizer(client_name=name).get_instance()

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=180.0,
        )

        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=180.0,
        )

        # Streaming control
        self._current_stop_event: asyncio.Event = asyncio.Event()
        self._current_output_queue: Optional[asyncio.Queue] = None
        self._current_stream_task: Optional[asyncio.Task] = None

        logger.info(
            f"LLMClient init | model={model} | base_url={self.base_url} | "
            f"extra_body_keys={list(self.extra_body_default)}"
        )

    def stop_stream(self) -> bool:
        """Request to interrupt only the active streaming call."""
        if not self._current_stop_event.is_set():
            self._current_stop_event.set()
            logger.debug(f"stop_stream() called on model={self.model} | event_set=True")
            return True
        logger.debug(f"stop_stream() called on model={self.model} | already stopping")
        return False

    def request_cancel_all(self):
        self._global_cancel_requested = True
        self.stop_stream()

    # -------------------------------------------------------------------------
    # Streaming chat – cancellable
    # -------------------------------------------------------------------------

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> asyncio.Queue:
        self.stop_stream()

        queue = asyncio.Queue()
        stop_event = asyncio.Event()

        self._current_output_queue = queue
        self._current_stop_event = stop_event

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        if messages:
            messages = prepare_chat_messages(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": True,
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        async def _run_stream():
            try:
                stream = await self.client.chat.completions.create(**params)
                async for chunk in stream:
                    if stop_event.is_set():
                        logger.debug("Stream stopped by stop_stream()")
                        break
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    await queue.put(content)
                await queue.put(None)
            except asyncio.CancelledError:
                await queue.put("[cancelled]")
                await queue.put(None)
            except Exception as e:
                logger.error(f"stream_chat error: {e}", exc_info=True)
                await queue.put(f"[error] {str(e)}")
                await queue.put(None)
            finally:
                if self._current_stream_task is asyncio.current_task():
                    self._current_stream_task = None
                if self._current_stop_event is stop_event:
                    self._current_stop_event = asyncio.Event()
                if self._current_output_queue is queue:
                    self._current_output_queue = None

        task = asyncio.create_task(_run_stream())
        self._current_stream_task = task

        return queue

    async def fetch_embedding(self, text: str, size: int = 500) -> np.ndarray | None:
        """
        Asynchronously fetches and caches an embedding for the given text.
        Automatically chunks long inputs and averages the results.
        Returns np.ndarray or None if embedding fails.
        """
        try:
            logger.info(f"Fetching embedding | input chars: {len(text)}")

            if not text.strip():
                logger.info("Empty input → returning None")
                return None

            # Use tokenizer if available, else treat whole text as single chunk
            if self.tokenizer:
                chunks = self.tokenizer.split_text_into_chunks(text, size)
            else:
                logger.warning(
                    "Tokenizer not available, treating entire text as a single chunk"
                )
                chunks = [text]

            logger.debug(f"Processing {len(chunks)} chunk(s)")

            if not chunks:
                return None

            # Embed all chunks
            chunk_embeddings: List[np.ndarray] = []
            for chunk in chunks:
                emb = await embed_single_chunk(self.client, chunk, self.model)
                if emb is not None:
                    chunk_embeddings.append(emb)

            if not chunk_embeddings:
                logger.error("All chunk embeddings failed")
                return None

            # Average pooling + L2 normalization
            stack = np.stack(chunk_embeddings)
            mean_vec = np.mean(stack, axis=0)
            norm = np.linalg.norm(mean_vec)

            if norm > 1e-9:
                mean_vec /= norm

            logger.debug(
                f"Final embedding dimension: {len(mean_vec)} "
                f"(averaged from {len(chunk_embeddings)} chunk(s))"
            )

            return mean_vec.astype(np.float32)

        except Exception as e:
            logger.error(
                f"Error fetching embedding for text (length {len(text)}): {str(e)}"
            )
            return None

    # -------------------------------------------------------------------------
    # Non-streaming chat (async + sync)
    # -------------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        # Make sure messages content is always str
        if messages:
            messages = prepare_chat_messages(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,  # always non-streaming
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = await self.client.chat.completions.create(**params)
            # Type safe cast to str
            return str((resp.choices[0].message.content or "")).strip()
        except Exception as e:
            logger.warning(
                f"chat failed | model={self.model}, msgs={len(messages)} | {str(e)[:180]}"
            )
            return f"[error] {str(e)}"

    # -------------------------------------------------------------------------
    # Describe image (vision)
    # -------------------------------------------------------------------------

    async def describe_image(
        self,
        image_base64: str | None,
        prompt: str = "Describe this image in detail.",
        vision_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        extra_body: Optional[Dict] = None,
        **kwargs: Any,
    ) -> str:
        """
        Describe an image using a vision-capable model.
        image_base64: base64-encoded image (jpeg/png/...).
        Uses the same base_url as the rest of the client.
        """
        if not image_base64:
            return "No image provided"

        model_to_use = vision_model or self.model

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ]

        if messages:
            messages = prepare_chat_messages(messages)

        # logger.debug(f"Describing image | tokens ≈ {count_message_tokens(messages)}")

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = await self.client.chat.completions.create(**params)
            description = (resp.choices[0].message.content or "").strip()
            return description if description else "No description generated"
        except Exception as e:
            logger.error(f"describe_image failed: {e}", exc_info=True)
            return f"[vision error] {str(e)}"

    async def call_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Union[List[Dict[str, Any]], "ActionSet"]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_choice: Union[
            str, Dict
        ] = "auto",  # "auto", "required", "none" or {"type": "function", "function": {"name": "..."}}
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        """
        Perform non-streaming tool calling.

        Accepts:
        - tools: List[dict]          → raw OpenAI tools schema
        - tools: ActionSet           → automatically uses .get_llm_tools()
        - tools: None                → no tools

        Returns:
        - List of tool calls (OpenAI format) or None
        """
        # Normalize tools to OpenAI format
        tool_schemas: Optional[List[Dict]] = None
        if tools is not None:
            if isinstance(tools, list):
                tool_schemas = tools
            elif isinstance(tools, ActionSet):
                tool_schemas = tools.get_llm_tools()
            else:
                raise TypeError("tools must be List[dict] or ActionSet")

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        if messages:
            messages = prepare_chat_messages(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "tool_choice": tool_choice,
            **kwargs,
        }

        if tool_schemas:
            params["tools"] = tool_schemas

        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = await self.client.chat.completions.create(**params)
            tool_calls = resp.choices[0].message.tool_calls
            if tool_calls:
                logger.info(
                    f"Tool calls returned: {[tc.function.name for tc in tool_calls]}"
                )
            return tool_calls
        except Exception as e:
            logger.error(f"call_tools failed: {e}", exc_info=True)
            return None

    async def stream_with_tools(
        self,
        manager,
        messages: List[Dict[str, Any]],
        tools: Optional[Union[List[Dict[str, Any]], "ActionSet"]] = None,
        temperature: float = 0.7,
        max_tokens: int = 22048,
        tool_choice: Union[str, Dict] = "auto",
        auto_stop_on_complete_tool: bool = False,
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> asyncio.Queue:
        """
        Streaming with tool support using ActionSet for executor resolution.

        - tools can be raw OpenAI tool schemas OR an ActionSet
        - Agent binding is handled by DynamicActionManager (set at init time)
        - Fully supports ConfirmationRequired exceptions
        """

        # ── Normalize tools ─────────────────────────────
        tool_schemas: Optional[List[Dict]] = None
        action_set: Optional[ActionSet] = None

        if not tools:
            tools = manager.get_action_set("DynamicCore")

        if tools is not None:
            if isinstance(tools, list):
                tool_schemas = tools
            elif isinstance(tools, ActionSet):
                action_set = tools
                tool_schemas = tools.get_llm_tools()

        queue: asyncio.Queue = asyncio.Queue()
        stop_event = asyncio.Event()
        self._current_stop_event = stop_event
        self._current_output_queue = queue

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        if messages:
            messages = prepare_chat_messages(messages)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "tool_choice": tool_choice,
            **kwargs,
        }
        if tool_schemas:
            params["tools"] = tool_schemas
        if merged_extra:
            params["extra_body"] = merged_extra

        # ── State for tool call streaming ─────────────────────────────
        tool_call_buffer: Dict[int, Dict] = {}
        tool_meta: Dict[int, Dict] = {}
        last_chunk = None

        # ── Streaming task ─────────────────────────────
        async def _run_stream():
            nonlocal last_chunk
            try:
                stream = await self.client.chat.completions.create(**params)

                async for chunk in stream:
                    if stop_event.is_set():
                        break

                    last_chunk = chunk
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # ── CONTENT STREAM ──────────────────────────────
                    if delta.content is not None:
                        await queue.put(("content", delta.content))

                    # ── TOOL CALLS STREAMING ────────────────────────
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            # Initialize buffers
                            if idx not in tool_call_buffer:
                                tool_call_buffer[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if idx not in tool_meta:
                                tool_meta[idx] = {
                                    "phase": "buffering",
                                    "last_len": 0,
                                    "completed": False,
                                }

                            tc = tool_call_buffer[idx]
                            meta = tool_meta[idx]

                            # Accumulate data
                            if tc_delta.id:
                                tc["id"] = tc_delta.id
                            if tc_delta.function.name:
                                tc["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

                            # Skip if we don't have a name yet or already completed
                            if not tc["function"]["name"] or meta["completed"]:
                                continue

                            # ── Incremental tool argument streaming ──
                            args = tc["function"]["arguments"]
                            new_len = len(args)
                            if new_len > meta["last_len"]:
                                chunk_delta = args[meta["last_len"] : new_len]
                                meta["last_len"] = new_len
                                await queue.put(
                                    (
                                        "tool_delta",
                                        {
                                            "index": idx,
                                            "id": tc["id"],
                                            "name": tc["function"]["name"],
                                            "arguments_delta": chunk_delta,
                                        },
                                    )
                                )

                            # ── COMPLETE when arguments form valid JSON ──
                            if is_valid_json(args) and not meta["completed"]:
                                meta["completed"] = True
                                payload = {"index": idx, **tc}

                                # Resolve and execute tool
                                if action_set is not None:
                                    try:
                                        # Use manager.get_executor (agent is already bound at load time)

                                        if manager is not None and hasattr(
                                            manager, "get_executor"
                                        ):
                                            action = manager.get_executor(
                                                name=tc["function"]["name"]
                                            )
                                        else:
                                            # fallback
                                            action = action_set.get_executor(
                                                name=tc["function"]["name"]
                                            )

                                        if action is None:
                                            raise RuntimeError(
                                                f"No executor found for tool '{tc['function']['name']}'"
                                            )

                                        args_dict = json.loads(args)
                                        result = action(**args_dict)

                                        if asyncio.iscoroutine(result) or isinstance(
                                            result, Awaitable
                                        ):
                                            result = await result

                                        payload["result"] = result

                                    except ConfirmationRequired as exc:
                                        # Forward confirmation request to consumer
                                        await queue.put(
                                            (
                                                "needs_confirmation",
                                                {
                                                    "index": idx,
                                                    "tool_name": tc["function"]["name"],
                                                    "details": exc.__dict__,
                                                },
                                            )
                                        )
                                        continue

                                    except Exception as e:
                                        logger.error(
                                            f"Tool execution error for '{tc['function']['name']}': {e}",
                                            exc_info=True,
                                        )
                                        payload["error"] = True
                                        payload["result"] = str(e)
                                else:
                                    # Fallback when only raw schemas were provided
                                    payload["result"] = None

                                await queue.put(("tool_complete", payload))

                                if auto_stop_on_complete_tool:
                                    break

                # ── STREAM FINISHED ─────────────────────────────
                final_tool_calls = (
                    [
                        {"index": i, **tool_call_buffer[i]}
                        for i in sorted(tool_call_buffer.keys())
                    ]
                    if tool_call_buffer
                    else None
                )

                finish_reason = (
                    last_chunk.choices[0].finish_reason
                    if last_chunk and last_chunk.choices
                    else "tool_calls"
                    if final_tool_calls
                    else "stop"
                )

                await queue.put(
                    (
                        "finish",
                        {
                            "finish_reason": finish_reason,
                            "tool_calls": final_tool_calls,
                        },
                    )
                )

            except asyncio.CancelledError:
                await queue.put(("finish", {"finish_reason": "cancelled"}))
            except Exception as e:
                logger.error(f"Stream error: {e}", exc_info=True)
                await queue.put(("error", str(e)))
            finally:
                await queue.put(None)
                if self._current_stop_event is stop_event:
                    self._current_stop_event = asyncio.Event()
                if self._current_output_queue is queue:
                    self._current_output_queue = None

        self._current_stream_task = asyncio.create_task(_run_stream())
        return queue

    # Tiny helper — makes the loop cleaner
    async def _drain_queue(
        self, queue: asyncio.Queue
    ) -> AsyncIterator[Tuple[str, Any]]:
        stop_event = self._current_stop_event  # always exists now

        while True:
            if stop_event.is_set():
                yield ("cancelled", "Stream cancelled")
                return

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            if item is None:
                break

            yield item  # or unpack if you prefer

            if stop_event.is_set():
                yield ("cancelled", "Stream cancelled")
                return

    # -------------------------------------------------------------------------
    # Quick single-turn convenience
    # -------------------------------------------------------------------------

    async def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, asyncio.Queue]:
        messages = prepare_chat_messages(user_content=prompt, system_prompt=system)

        if stream:
            # Return the raw streaming queue
            return await self.stream_chat(messages, **kwargs)

        # Non-streaming: drain queue into a string
        queue = await self.stream_chat(messages, **kwargs)
        return await drain_queue_to_string(queue)

    def ask_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, asyncio.Queue]:

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop → safe
            return asyncio.run(self.ask(prompt, system=system, stream=stream, **kwargs))

        # Already running loop → thread fallback
        import threading

        result_container: dict[str, Any] = {
            "result": None,
            "error": None,
        }

        def runner():
            try:
                result_container["result"] = asyncio.run(
                    self.ask(prompt, system=system, stream=stream, **kwargs)
                )
            except Exception as e:
                result_container["error"] = e

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if result_container["error"]:
            raise result_container["error"]

        return result_container["result"]
