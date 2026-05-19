"""
Mock LLM Proxy Server — OpenAI-compatible streaming + non-streaming.

This server acts as a controllable proxy: the test orchestrator enqueues
response payloads (including tool calls) via `enqueue_response()`, and the
server replays them in standard OpenAI SSE / JSON format.

No prompts are hardcoded. Every response is fed externally by the test.
Integrated from advanced tests into utils for auto-deployment via config.
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web


class MockLLMServer:
    """An aiohttp-based OpenAI-compatible mock server.

    Supports two usage modes:
    - Async: await server.start() / await server.stop()  (same event loop)
    - Sync auto-deploy: server.start_sync()  (background thread + loop)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9111):
        self.host = host
        self.port = port
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self.request_log: List[Dict[str, Any]] = []

        # For sync (background thread) mode
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def enqueue_response(self, response: Dict[str, Any]) -> None:
        """Push a response dict the server will use for the next request.

        For a plain text response:
            {"content": "Hello there!"}

        For a tool-call response:
            {"tool_calls": [{"name": "my_tool", "arguments": {"arg": "val"}}]}

        For combined (content + tool_calls):
            {"content": "Let me call...", "tool_calls": [...]}
        """
        self._response_queue.put_nowait(response)

    async def _get_next_response(self, timeout: float = 15.0) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(self._response_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {"content": "[mock-server] No response enqueued (timeout)"}

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    async def _handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse:
        body = await request.json()
        self.request_log.append(body)

        stream = body.get("stream", False)
        resp_data = await self._get_next_response()

        if stream:
            return await self._stream_response(request, body, resp_data)
        else:
            return self._non_stream_response(body, resp_data)

    def _non_stream_response(self, body: Dict, resp_data: Dict) -> web.Response:
        content = resp_data.get("content", "")
        tool_calls_raw = resp_data.get("tool_calls")

        message: Dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"

        if tool_calls_raw:
            finish_reason = "tool_calls"
            tc_list = []
            for i, tc in enumerate(tool_calls_raw):
                tc_list.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                        "index": i,
                    }
                )
            message["tool_calls"] = tc_list

        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        return web.json_response(payload)

    async def _stream_response(
        self, request: web.Request, body: Dict, resp_data: Dict
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = body.get("model", "mock-model")
        content = resp_data.get("content", "")
        tool_calls_raw = resp_data.get("tool_calls")

        # Stream content tokens
        if content:
            words = content.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0.005)

        # Stream tool calls
        if tool_calls_raw:
            for i, tc in enumerate(tool_calls_raw):
                tc_id = f"call_{uuid.uuid4().hex[:12]}"
                args_str = json.dumps(tc.get("arguments", {}))

                # First chunk: id + function name
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc_id,
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0.005)

                # Stream arguments in small chunks for realistic streaming
                chunk_size = max(1, len(args_str) // 3)
                for pos in range(0, len(args_str), chunk_size):
                    arg_chunk = args_str[pos : pos + chunk_size]
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": i,
                                            "function": {
                                                "arguments": arg_chunk,
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    await asyncio.sleep(0.005)

        # Final chunk
        finish_reason = "tool_calls" if tool_calls_raw else "stop"
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
        await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        body = await request.json()
        dim = 384
        embedding = [0.01] * dim
        payload = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
            "model": body.get("model", "mock-embed"),
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        return web.json_response(payload)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
        self._app.router.add_post("/v1/embeddings", self._handle_embeddings)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        # If this instance owns a background loop, stop it too
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------
    # Sync deployment helpers (for ConfigLoader auto :test server)
    # ------------------------------------------------------------------
    def start_sync(self) -> None:
        """Start the server in a background daemon thread (own event loop).

        Ideal for ConfigLoader auto-deployment when `server: {type: test}` is present.
        The thread is daemon so it won't block process exit.
        """
        if self._server_thread and self._server_thread.is_alive():
            return

        def _runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self.start())
            try:
                self._loop.run_forever()
            except Exception:
                pass
            finally:
                if not self._loop.is_closed():
                    self._loop.close()

        self._server_thread = threading.Thread(
            target=_runner, daemon=True, name=f"MockLLMServer-{self.port}"
        )
        self._server_thread.start()
        # Allow the server to bind before returning
        time.sleep(0.3)

    def stop_sync(self) -> None:
        """Best-effort shutdown for sync-started server."""
        if self._loop and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self.stop(), self._loop)
                fut.result(timeout=3.0)
            except Exception:
                pass
        # Thread will exit because daemon + loop stopped
