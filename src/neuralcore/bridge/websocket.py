import asyncio
import json
from typing import Any, Optional

from websockets.asyncio.server import ServerConnection, serve

from neuralcore.agents.core import Agent
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class WebSocketBridge:
    """
    Rich WebSocket bridge for live agent telemetry (NeuralCore).
    Exposes full AgentState including waiting, sub-tasks, workflow events, etc.
    """

    def __init__(self, agent: Agent, host: str = "127.0.0.1", port: int = 8765):
        self.agent = agent
        self.host = host
        self.port = port

        self._server = None
        self._clients: set[ServerConnection] = set()
        self._state_broadcast_task: Optional[asyncio.Task] = None

        # Hook into background events
        self.agent.on_background_event = self._on_background_event

    async def _on_background_event(self, event: str, payload: Any):
        """Forward background events with full state (including waiting)."""
        message = {
            "type": "background_event",
            "event": event,
            "payload": payload
            if isinstance(payload, dict)
            else {"content": str(payload)},
            "agent_id": self.agent.agent_id,
            "full_state": self.agent.get_full_state_dict(),
            "timestamp": asyncio.get_event_loop().time(),
        }
        await self._broadcast(message)

    async def _broadcast(self, message: dict):
        """Broadcast to all connected clients, removing dead ones."""
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps(message))
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _state_heartbeat(self):
        """Broadcast state updates regularly — including when waiting."""
        while True:
            try:
                # Always send heartbeat when agent is active or waiting
                if self.agent.status in (
                    "running",
                    "thinking",
                    "tool_call",
                    "execution",
                    "waiting",
                ) or getattr(self.agent.state, "waiting", False):
                    await self._broadcast(
                        {
                            "type": "state_update",
                            "data": self.agent.get_full_state_dict(),
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )
                await asyncio.sleep(0.8)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"WS heartbeat error: {e}")

    async def _handler(self, websocket: ServerConnection):
        self._clients.add(websocket)
        logger.info(f"[WS] Client connected → {self.agent.name}")

        # Start heartbeat if not already running
        if not self._state_broadcast_task or self._state_broadcast_task.done():
            self._state_broadcast_task = asyncio.create_task(self._state_heartbeat())

        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )
                    continue

                cmd = msg.get("command")

                if cmd == "send":
                    await self.agent.post_message(msg.get("content", ""))
                elif cmd == "system":
                    await self.agent.post_system_message(msg.get("content", ""))
                elif cmd == "control":
                    await self.agent.post_control(msg.get("payload", {}))
                elif cmd == "status":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "status",
                                "data": await self.agent.get_agent_status(),
                            }
                        )
                    )
                elif cmd == "full_state":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "full_state",
                                "data": self.agent.get_full_state_dict(),
                            }
                        )
                    )
                elif cmd == "get_sub_tasks":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "sub_tasks",
                                "data": self.agent.get_sub_tasks(),
                            }
                        )
                    )
                elif cmd == "cancel_sub_task":
                    task_id = msg.get("task_id")
                    success = self.agent.cancel_sub_task(task_id) if task_id else False
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "ack",
                                "action": "cancel_sub_task",
                                "success": success,
                            }
                        )
                    )
                elif cmd == "stop":
                    if self.agent._stop_event:
                        self.agent._stop_event.set()
                    await websocket.send(json.dumps({"type": "ack", "action": "stop"}))
                elif cmd == "reload_config":
                    self.agent.reload_config(msg.get("config"))
                    await websocket.send(
                        json.dumps({"type": "ack", "action": "reload_config"})
                    )
                else:
                    await websocket.send(
                        json.dumps(
                            {"type": "error", "message": f"Unknown command: {cmd}"}
                        )
                    )
        finally:
            self._clients.discard(websocket)
            logger.info(f"[WS] Client disconnected from {self.agent.name}")

    async def start(self):
        self._server = await serve(self._handler, self.host, self.port)
        logger.info(f"🚀 WebSocketBridge live → ws://{self.host}:{self.port}")
        await self._server.serve_forever()

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._state_broadcast_task:
            self._state_broadcast_task.cancel()
        logger.info("WebSocketBridge stopped")
