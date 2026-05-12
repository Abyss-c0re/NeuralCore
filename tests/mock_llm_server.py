"""
Mock LLM Server -- OpenAI-compatible API for NeuralCore testing.

Implements /v1/chat/completions with:
- Non-streaming responses
- SSE streaming responses (compatible with openai.AsyncOpenAI)
- Tool call streaming (OpenAI format)
- Programmable response engine (no hardcoded prompts)
- Message analysis for context-appropriate responses

Compatible with: openai>=1.0 AsyncOpenAI client
"""

import asyncio
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
from aiohttp import web


class ResponseEngine:
    """Programmable response engine -- analyzes messages to produce responses.

    Supports registering custom handlers for specific patterns.
    Falls back to intelligent default behavior based on message analysis.
    No hardcoded prompts -- all responses are derived from message content.
    """

    def __init__(self):
        self._handlers: List[tuple] = []
        self._tool_handlers: List[tuple] = []
        self._default_response = "I understand your request. Let me help with that."

    def register_handler(self, matcher: Callable, responder: Callable):
        self._handlers.append((matcher, responder))

    def register_tool_handler(self, matcher: Callable, responder: Callable):
        self._tool_handlers.append((matcher, responder))

    def generate_response(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Analyze messages and produce a response dict with content and/or tool_calls."""
        # Custom tool handlers
        if tools:
            for matcher, responder in self._tool_handlers:
                try:
                    if matcher(messages, tools):
                        tc = responder(messages, tools)
                        return {"content": None, "tool_calls": tc}
                except Exception:
                    continue

        # Custom text handlers
        for matcher, responder in self._handlers:
            try:
                if matcher(messages):
                    return {"content": responder(messages), "tool_calls": None}
            except Exception:
                continue

        return self._default_analyze(messages, tools)

    def _default_analyze(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Intelligent default response based on message content analysis."""
        last_user = self._get_last_user_message(messages)
        all_text = " ".join(
            m.get("content", "") or ""
            for m in messages
            if isinstance(m.get("content"), str)
        )
        lower = all_text.lower()

        # Intent classification
        if "classify" in lower and ("casual" in lower or "task" in lower):
            if any(
                w in last_user.lower()
                for w in ["hello", "hi", "hey", "thanks", "joke", "how are"]
            ):
                return {"content": "CASUAL", "tool_calls": None}
            return {"content": "TASK", "tool_calls": None}

        # Simple/complex classification
        if "simple" in lower and "complex" in lower and "exactly one word" in lower:
            return {"content": "SIMPLE", "tool_calls": None}

        # Task decomposition / planning
        if (
            "task decomposition" in lower
            or "break this request" in lower
            or "actionable steps" in lower
            or "minimal number of clear" in lower
        ):
            return {"content": self._generate_plan(last_user), "tool_calls": None}

        # Validation (YES/NO)
        if (
            "validation" in lower
            or "has the expected outcome" in lower
            or "fully achieved" in lower
        ):
            return {"content": "YES", "tool_calls": None}

        # Tool result messages (after tool execution) -- complete the task
        if any(m.get("role") == "tool" for m in messages):
            return {
                "content": "Task completed successfully. [FINAL_ANSWER_COMPLETE]",
                "tool_calls": None,
            }

        # Tool calling -- if tools are provided and action is needed
        if tools and self._should_use_tools(last_user, tools):
            return self._generate_tool_call(messages, tools)

        # JSON response requests
        if "json" in lower and ("topic_name" in lower or "topic_description" in lower):
            return {
                "content": json.dumps(
                    {
                        "topic_name": "General",
                        "topic_description": "General conversation",
                    }
                ),
                "tool_calls": None,
            }

        # Multi-query generation
        if "search queries" in lower and "json array" in lower:
            return {
                "content": json.dumps(["query one", "query two", "query three"]),
                "tool_calls": None,
            }

        # Default chat response
        return {"content": self._contextual_reply(last_user), "tool_calls": None}

    def _get_last_user_message(self, messages: List[Dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    )
        return ""

    def _generate_plan(self, query: str) -> str:
        return json.dumps(
            {
                "steps": [
                    {
                        "description": f"Execute: {query[:100]}",
                        "dependencies": [],
                        "suggested_tool": "",
                        "expected_outcome": "Task completed successfully",
                    }
                ]
            }
        )

    def _should_use_tools(self, message: str, tools: List[Dict]) -> bool:
        action_words = [
            "read",
            "write",
            "search",
            "find",
            "create",
            "execute",
            "list",
            "run",
            "analyze",
        ]
        msg_lower = message.lower()
        return any(w in msg_lower for w in action_words) and len(tools) > 0

    def _generate_tool_call(
        self, messages: List[Dict], tools: List[Dict]
    ) -> Dict[str, Any]:
        last_msg = self._get_last_user_message(messages)
        msg_lower = last_msg.lower()

        best_tool = None
        best_score = 0
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "").lower()
            desc = fn.get("description", "").lower()
            score = sum(1 for w in msg_lower.split() if w in name or w in desc)
            if score > best_score:
                best_score = score
                best_tool = t

        if not best_tool or best_score == 0:
            best_tool = tools[0]

        fn = best_tool["function"]
        params = fn.get("parameters", {})
        required = params.get("required", [])
        args = {}
        for pname in required:
            prop = params.get("properties", {}).get(pname, {})
            ptype = prop.get("type", "string")
            if ptype == "string":
                args[pname] = last_msg[:200]
            elif ptype in ("integer", "number"):
                args[pname] = 1
            elif ptype == "boolean":
                args[pname] = True

        return {
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": fn["name"], "arguments": json.dumps(args)},
                }
            ],
        }

    def _contextual_reply(self, message: str) -> str:
        if not message.strip():
            return self._default_response
        return (
            f"Understood. Processing your request regarding: "
            f"{message[:120]}. Task complete. [FINAL_ANSWER_COMPLETE]"
        )


class MockLLMServer:
    """OpenAI-compatible mock LLM server for testing.

    Uses aiohttp to serve /v1/chat/completions, /v1/embeddings, /v1/models.
    Streaming uses proper SSE format compatible with the openai Python library.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9111):
        self.host = host
        self.port = port
        self.engine = ResponseEngine()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self.request_log: List[Dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    async def start(self):
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._app.router.add_post("/v1/embeddings", self._handle_embeddings)
        self._app.router.add_get("/v1/models", self._handle_models)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def stop(self):
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    async def _handle_models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [{"id": "mock-model", "object": "model", "owned_by": "test"}],
            }
        )

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        import numpy as np

        body = await request.json()
        dim = 384
        seed = hash(str(body.get("input", ""))) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(float)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": vec.tolist(), "index": 0}
                ],
                "model": body.get("model", "mock-embed"),
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )

    async def _handle_chat(
        self, request: web.Request
    ) -> Union[web.Response, web.StreamResponse]:
        body = await request.json()
        self.request_log.append(
            {
                "timestamp": time.time(),
                "model": body.get("model"),
                "stream": body.get("stream", False),
                "message_count": len(body.get("messages", [])),
                "has_tools": bool(body.get("tools")),
            }
        )

        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        model = body.get("model", "mock-model")

        result = self.engine.generate_response(messages, tools)
        content = result.get("content")
        tool_calls = result.get("tool_calls")

        if stream:
            return await self._stream_response(request, content, tool_calls, model)
        else:
            return self._non_stream_response(content, tool_calls, model)

    def _non_stream_response(
        self, content: Optional[str], tool_calls: Optional[List[Dict]], model: str
    ) -> web.Response:
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = "tool_calls" if tool_calls else "stop"

        return web.json_response(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": max(len(content or "") // 4, 10),
                    "total_tokens": 60 + max(len(content or "") // 4, 10),
                },
            }
        )

    async def _stream_response(
        self,
        request: web.Request,
        content: Optional[str],
        tool_calls: Optional[List[Dict]],
        model: str,
    ) -> web.StreamResponse:
        """SSE streaming response compatible with openai.AsyncOpenAI."""
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        async def _write_chunk(data: dict):
            await resp.write(f"data: {json.dumps(data)}\n\n".encode())

        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")

                # First chunk: tool call id + name
                await _write_chunk(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "index": tc_idx,
                                            "id": tc.get(
                                                "id", f"call_{uuid.uuid4().hex[:8]}"
                                            ),
                                            "type": "function",
                                            "function": {
                                                "name": fn.get("name", ""),
                                                "arguments": "",
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                await asyncio.sleep(0.005)

                # Stream arguments in chunks
                chunk_size = max(10, len(args_str) // 3)
                for i in range(0, len(args_str), chunk_size):
                    await _write_chunk(
                        {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": tc_idx,
                                                "function": {
                                                    "arguments": args_str[
                                                        i : i + chunk_size
                                                    ]
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    await asyncio.sleep(0.005)

            # Finish
            await _write_chunk(
                {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                }
            )

        elif content:
            words = content.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                delta = {"content": token}
                if i == 0:
                    delta["role"] = "assistant"
                await _write_chunk(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                await asyncio.sleep(0.002)

            await _write_chunk(
                {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        else:
            await _write_chunk(
                {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        await resp.write(b"data: [DONE]\n\n")
        return resp


if __name__ == "__main__":

    async def main():
        server = MockLLMServer(port=9111)
        await server.start()
        print(f"Mock LLM server running at {server.base_url}")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await server.stop()

    asyncio.run(main())
