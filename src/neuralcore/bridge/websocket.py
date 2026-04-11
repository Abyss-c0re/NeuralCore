import asyncio
import json
from typing import Any

from websockets.asyncio.server import ServerConnection, serve


from neuralcore.agents.core import Agent
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class WebSocketBridge:
    """
    Lightweight local WebSocket bridge for bidirectional communication with agents.
    """

    def __init__(
        self,
        agent: Agent,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.agent = agent
        self.host = host
        self.port = port

        self._server = None
        self._clients: set[ServerConnection] = set()

        self.agent.on_background_event = self._on_background_event

    async def _on_background_event(self, event: str, payload: Any):
        """Forward every background event to connected clients."""
        message = {
            "type": "background_event",
            "event": event,
            "payload": payload
            if isinstance(payload, dict)
            else {"content": str(payload)},
            "agent_id": self.agent.agent_id,
            "timestamp": asyncio.get_event_loop().time(),
        }

        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(json.dumps(message))
            except Exception:
                dead.add(ws)
        self._clients -= dead

    async def _handler(self, websocket: ServerConnection):
        self._clients.add(websocket)
        logger.info(f"[WS] Client connected to headless agent '{self.agent.name}'")

        try:
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                    cmd = msg.get("command")

                    if cmd == "send":
                        await self.agent.post_message(msg.get("content", ""))
                    elif cmd == "system":
                        await self.agent.post_system_message(msg.get("content", ""))
                    elif cmd == "control":
                        await self.agent.post_control(msg.get("payload", {}))
                    elif cmd == "status":
                        status = await self.agent.get_agent_status()
                        await websocket.send(
                            json.dumps({"type": "status", "data": status})
                        )
                    elif cmd == "stop":
                        if self.agent._stop_event:
                            self.agent._stop_event.set()
                        await websocket.send(
                            json.dumps({"type": "ack", "action": "stop"})
                        )
                except Exception as e:
                    await websocket.send(
                        json.dumps({"type": "error", "message": str(e)})
                    )
        finally:
            self._clients.discard(websocket)
            logger.info(f"[WS] Client disconnected from '{self.agent.name}'")

    async def start(self):
        """Start the WebSocket server."""
        self._server = await serve(
            self._handler,
            self.host,
            self.port,
        )
        logger.info(
            f"🚀 WebSocket bridge running for '{self.agent.name}' → ws://{self.host}:{self.port}"
        )
        await self._server.serve_forever()

    async def stop(self):
        """Graceful shutdown."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("WebSocket bridge stopped")
