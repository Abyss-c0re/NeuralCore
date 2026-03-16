import asyncio
from typing import List, Dict, Any, Optional, Union
import json
import tiktoken
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from src.neuralcore.actions.actions import Action, ActionSet
from src.neuralcore.actions.manager import DynamicActionManager
from src.neuralcore.utils.exceptions_handler import ConfirmationRequired
from typing import AsyncIterator, Callable, Tuple

import numpy as np

from src.neuralcore.utils.logger import Logger

ToolProvider = Union[ActionSet, DynamicActionManager, List[Dict[str, Any]]]
ToolExecutorGetter = Optional[Callable[[str], Optional["Action"]]]


logger = Logger.get_logger()

MAX_CHUNK_TOKENS = 900  # safety margin below 1024
OVERLAP_TOKENS = 100

# Rough tokenizer - cl100k_base is usually close enough for modern models
_tokenizer = tiktoken.get_encoding("cl100k_base")


def split_text_into_chunks(
    text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS
) -> List[str]:
    """Split text into overlapping chunks that fit within max_tokens."""
    if not text.strip():
        return []

    tokens = _tokenizer.encode(text, allowed_special="all")
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start_idx += max_tokens - overlap

    return chunks


async def embed_single_chunk(
    client: OpenAI,  # sync client
    chunk: str,
    model: str,
) -> Optional[np.ndarray]:
    try:
        response = await asyncio.to_thread(
            client.embeddings.create, model=model, input=chunk.strip()
        )
        vec = response.data[0].embedding
        return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error(f"Embedding chunk failed: {str(e)[:180]}...")
        return None


def count_message_tokens(messages: List[dict]) -> int:
    """Approximate token count for a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_tokenizer.encode(content))
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    total += len(_tokenizer.encode(item.get("text", "")))
    return total  # Approximate, ignores role tokens etc.


@staticmethod
def build_messages(
    user_content: Optional[Union[str, List[Dict]]] = None,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    if history is not None:
        return history[:]
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    if user_content:
        msgs.append({"role": "user", "content": user_content})
    return msgs


class LLMClient:
    """
    Universal OpenAI-compatible client with:
    - cancellable streaming via stop_stream()
    - both async & sync support
    - image description (vision) support using the same base_url
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        extra_body: Optional[Dict[str, Any]] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.extra_body_default = extra_body or {}

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
        self._current_stop_event: Optional[asyncio.Event] = None
        self._current_output_queue: Optional[asyncio.Queue] = None
        self._current_stream_task: Optional[asyncio.Task] = None

        logger.info(
            f"LLMClient init | model={model} | base_url={self.base_url} | "
            f"extra_body_keys={list(self.extra_body_default)}"
        )

    def stop_stream(self) -> bool:
        """Request to interrupt any active streaming call."""
        if self._current_stop_event is not None:
            self._current_stop_event.set()
            logger.debug("stop_stream() triggered")
            return True
        return False

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

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.default_max_tokens,
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
                    self._current_stop_event = None
                if self._current_output_queue is queue:
                    self._current_output_queue = None

        task = asyncio.create_task(_run_stream())
        self._current_stream_task = task

        return queue

    async def fetch_embedding(self, text: str) -> np.ndarray | None:
        """
        Asynchronously fetches and caches an embedding for the given text.
        Automatically chunks long inputs and averages the results.
        Returns plain list[float] to stay compatible with original behavior.
        """
        try:
            logger.info(f"Fetching embedding | input chars: {len(text)}")

            if not text.strip():
                logger.info("Empty input → returning None")
                return None

            # Split into chunks if necessary
            chunks = split_text_into_chunks(text)
            logger.debug(f"Processing {len(chunks)} chunk(s)")

            if not chunks:
                return None

            # Embed all chunks
            chunk_embeddings: List[np.ndarray] = []
            for chunk in chunks:
                emb = await embed_single_chunk(self.sync_client, chunk, self.model)
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

            embedding_list = mean_vec.tolist()

            logger.debug(
                f"Final embedding dimension: {len(embedding_list)} "
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

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.default_max_tokens,
            "stream": False,
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = await self.client.chat.completions.create(**params)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # Minimal logging
            logger.warning(
                f"chat failed | model={self.model}, msgs={len(messages)} | {str(e)[:180]}"
            )
            return f"[error] {str(e)}"

    def chat_sync(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.default_max_tokens,
            "stream": False,
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = self.sync_client.chat.completions.create(**params)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(
                f"chat_sync failed | model={self.model}, msgs={len(messages)} | {str(e)[:180]}"
            )
            return f"[sync error] {str(e)}"

    # -------------------------------------------------------------------------
    # Describe image (vision) – async + sync
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

        logger.debug(f"Describing image | tokens ≈ {count_message_tokens(messages)}")

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
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

    def describe_image_sync(
        self,
        image_base64: str | None,
        prompt: str = "Describe this image in detail.",
        vision_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
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

        merged_extra = {**self.extra_body_default, **kwargs.pop("extra_body", {})}

        params = {
            "model": model_to_use,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.default_temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }
        if merged_extra:
            params["extra_body"] = merged_extra

        try:
            resp = self.sync_client.chat.completions.create(**params)
            description = (resp.choices[0].message.content or "").strip()
            return description if description else "No description generated"
        except Exception as e:
            logger.error(f"describe_image_sync failed: {e}", exc_info=True)
            return f"[sync vision error] {str(e)}"

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
        messages: List[Dict[str, Any]],
        tools: Optional[Union[List[Dict[str, Any]], "ActionSet"]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        tool_choice: Union[str, Dict] = "auto",
        auto_stop_on_complete_tool: bool = False,  # default off — safer
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> asyncio.Queue:
        """
        Streaming with tool support — returns queue of tagged events

        Queue items are tuples: (kind: str, payload: Any)

        Possible kinds:
        - "content"         → str (text delta)
        - "tool_delta"      → dict (partial update of one tool call)
        - "tool_complete"   → dict (finished tool call - only if auto_stop=True)
        - "finish"          → dict { "finish_reason": str, "tool_calls": list|None }
        - "error"           → str (error message)
        - None              → end of stream marker

        Recommended usage pattern:
            text_buffer = ""
            tool_buffer = {}   # index → current tool call dict
            while True:
                item = await queue.get()
                if item is None: break
                kind, payload = item
                if kind == "content":
                    text_buffer += payload
                    # update UI
                elif kind == "tool_delta":
                    idx = payload["index"]
                    tool_buffer[idx] = payload
                    # optional: show partial tool call in UI
                elif kind == "finish":
                    # decide what to do — execute tools, append messages, etc.
        """
        # Normalize tools
        tool_schemas: Optional[List[Dict]] = None
        if tools is not None:
            if isinstance(tools, list):
                tool_schemas = tools
            elif isinstance(tools, ActionSet):
                tool_schemas = tools.get_llm_tools()
            else:
                raise TypeError("tools must be List[dict] or ActionSet")

        self.stop_stream()  # assuming you have this method

        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        self._current_stop_event = stop_event
        self._current_output_queue = queue

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

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

        tool_call_buffer: Dict[int, Dict] = {}  # index → current tool call state
        last_chunk = None

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

                    # ── Content ───────────────────────────────────────
                    if delta.content is not None:
                        await queue.put(("content", delta.content))

                    # ── Tool calls ────────────────────────────────────
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            if idx not in tool_call_buffer:
                                tool_call_buffer[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }

                            tc = tool_call_buffer[idx]

                            changed = False

                            if tc_delta.id:
                                tc["id"] += tc_delta.id
                                changed = True
                            if tc_delta.function.name:
                                tc["function"]["name"] += tc_delta.function.name
                                changed = True
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )
                                changed = True

                            if changed:
                                await queue.put(
                                    (
                                        "tool_delta",
                                        {
                                            "index": idx,
                                            **tc,  # current state of this tool call
                                        },
                                    )
                                )

                            # Optional early complete detection (use with caution)
                            if auto_stop_on_complete_tool:
                                args_str = tc["function"]["arguments"].strip()
                                if args_str and args_str[-1] in {"}", "]"}:
                                    try:
                                        json.loads(args_str)
                                        await queue.put(
                                            ("tool_complete", {"index": idx, **tc})
                                        )
                                        logger.debug(
                                            f"Tool call {tc['function']['name']} appears complete → stopping"
                                        )
                                        stop_event.set()
                                        break
                                    except json.JSONDecodeError:
                                        pass

                # ── Stream ended normally ─────────────────────────────
                final_tool_calls = None
                if tool_call_buffer:
                    final_tool_calls = [
                        {"index": i, **tool_call_buffer[i]}
                        for i in sorted(tool_call_buffer.keys())
                    ]

                finish_reason = None
                if last_chunk and last_chunk.choices:
                    finish_reason = last_chunk.choices[0].finish_reason

                # heuristic fallback
                if finish_reason is None:
                    finish_reason = "tool_calls" if final_tool_calls else "stop"

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
                logger.error(f"stream_with_tools failed: {e}", exc_info=True)
                await queue.put(("error", str(e)))
            finally:
                await queue.put(None)
                self._current_stop_event = None
                self._current_output_queue = None
                if self._current_stream_task is asyncio.current_task():
                    self._current_stream_task = None

        self._current_stream_task = asyncio.create_task(_run_stream())
        return queue

    def call_tools_sync(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[Union[List[Dict[str, Any]], "ActionSet"]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_choice: Union[str, Dict] = "auto",
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        """
        Synchronous (blocking) version of call_tools.

        Same input flexibility:
          - tools: List[dict]           → raw OpenAI tool schemas
          - tools: ActionSet            → uses .get_llm_tools()
          - tools: None                 → no tools

        Returns:
          List of tool calls or None
        """
        # Normalize tools → OpenAI-compatible schema
        tool_schemas: Optional[List[Dict]] = None
        if tools is not None:
            if isinstance(tools, list):
                tool_schemas = tools
            elif isinstance(tools, ActionSet):
                tool_schemas = tools.get_llm_tools()
            else:
                raise TypeError("tools must be List[dict] or ActionSet")

        merged_extra = {**self.extra_body_default, **(extra_body or {})}

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
            resp = self.sync_client.chat.completions.create(**params)
            tool_calls = resp.choices[0].message.tool_calls
            if tool_calls:
                logger.info(
                    f"[sync] Tool calls returned: {[tc.function.name for tc in tool_calls]}"
                )
            return tool_calls
        except Exception as e:
            logger.error(f"[sync] call_tools_sync failed: {e}", exc_info=True)
            return None

    async def agent_stream(
        self,
        user_prompt: str,
        tools: ToolProvider,
        messages_so_far: List[Dict[str, Any]],  # ← caller provides this
        system_prompt: str = "",
        get_executor: ToolExecutorGetter = None,
        history_manager=None,  # ← optional, passed in
        max_iterations: int = 25,
        temperature: float = 0.3,
        max_tokens: int = 12048,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Does **not** read self.conversation, self.history_manager, self.system_prompt
        """

        # Tool accessors (unchanged)
        if isinstance(tools, (ActionSet, DynamicActionManager)):
            get_tools = tools.get_llm_tools
            get_executor = get_executor or tools.get_executor
        else:
            get_tools = lambda: tools
            get_executor = get_executor or (lambda name: None)

        # Start with a copy — never mutate caller's list
        messages = messages_so_far.copy()

        # Add system prompt if not already present
        if system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Context enrichment — only if caller provides history_manager
        if history_manager is not None:
            try:
                context_history = await history_manager.generate_prompt(
                    user_prompt, num_messages=0
                )
                if (
                    context_history
                    and "\nUser query:" in context_history[-1]["content"]
                ):
                    ctx = (
                        context_history[-1]["content"].split("\nUser query:")[0].strip()
                    )
                    if ctx:
                        messages.insert(
                            1,
                            {"role": "system", "content": f"Relevant context:\n{ctx}"},
                        )
            except Exception as e:
                yield ("log", f"History enrichment failed: {e}")

        executed_signatures: set[tuple] = set()
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            yield ("step_start", {"iteration": iteration})

            queue = await self.stream_with_tools(
                messages=messages,
                tools=get_tools(),
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice="auto",
            )

            text_buffer = ""
            tool_calls = None

            async for kind, payload in self._drain_queue(queue):
                if kind == "content":
                    text_buffer += payload
                    yield ("content_delta", payload)

                elif kind == "tool_delta":
                    yield ("tool_call_delta", payload)

                elif kind == "finish":
                    tool_calls = payload.get("tool_calls")
                    yield ("llm_finish", payload)
                    break

                elif kind == "error":
                    yield ("error", payload)
                    return

            full_reply = text_buffer.strip()

            if not tool_calls:
                # Check if the last message is already an assistant message
                if messages and messages[-1]["role"] == "assistant":
                    # If it's the same assistant message, update its content
                    messages[-1]["content"] = full_reply
                else:
                    # Otherwise, add a new assistant message
                    yield (
                        "assistant_message",
                        {"role": "assistant", "content": full_reply},
                    )
                yield ("final_answer", full_reply)
                return

            # Handle tool calls and messages correctly
            messages.append({"role": "assistant", "tool_calls": tool_calls})
            yield ("tool_calls", tool_calls)

            # ── tool execution loop (unchanged from your version) ──
            tool_results = []
            for call in tool_calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"])
                except Exception:
                    args = {}
                    yield ("log", f"Failed to parse args for {name}")

                sig = (name, json.dumps(args, sort_keys=True))
                if sig in executed_signatures:
                    yield ("log", f"Skipping duplicate tool call: {name}")
                    continue

                executed_signatures.add(sig)

                yield ("tool_start", {"name": name, "args": args})

                action = get_executor(
                    name
                )  # ← DynamicActionManager / ActionSet decides
                if not action:
                    result = f"Unknown tool: {name}"
                    yield (
                        "tool_result",
                        {"name": name, "result": result, "error": True},
                    )
                else:
                    try:
                        maybe_result = action(**args)
                        result = (
                            await maybe_result
                            if asyncio.iscoroutine(maybe_result)
                            else maybe_result
                        )
                        yield ("tool_result", {"name": name, "result": result})
                    except ConfirmationRequired as exc:
                        # Keep the same confirmation flow — yield special event
                        yield (
                            "needs_confirmation",
                            {
                                "tool_call_id": call["id"],
                                "name": name,
                                "args": args,
                                "preview": exc.preview or "",
                                "action": action,
                                "tool_calls": tool_calls,  # so UI can resume later
                            },
                        )
                        return  # ← important: pause here, let UI/resumer continue
                    except Exception as exc:
                        result = f"Tool execution failed: {exc}"
                        yield (
                            "tool_result",
                            {"name": name, "result": result, "error": True},
                        )

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": str(result)
                        if not isinstance(result, str)
                        else result,
                    }
                )

            if tool_results:
                messages.extend(tool_results)
            else:
                yield ("warning", "No tool results produced")
                break

        if iteration >= max_iterations:
            yield ("finish", {"reason": "max_iterations_reached"})

    # Tiny helper — makes the loop cleaner
    async def _drain_queue(
        self, queue: asyncio.Queue
    ) -> AsyncIterator[Tuple[str, Any]]:
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, payload = item
            yield kind, payload

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
        messages = build_messages(user_content=prompt, system_prompt=system)
        if stream:
            return await self.stream_chat(messages, **kwargs)
        return await self.chat(messages, **kwargs)

    def ask_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = build_messages(user_content=prompt, system_prompt=system)
        return self.chat_sync(messages, **kwargs)
