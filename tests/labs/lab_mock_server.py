"""
Advanced LLM Mock Proxy Server for Neuroscience Cross-Domain RAG Experiment.

Extends the base MockLLMServer with a neuroscience-aware response engine that:
- Handles intent classification (CASUAL/TASK)
- Handles planning / task decomposition
- Generates concept extraction JSON
- Produces neuroscience-rich responses from context
- Handles multi-query generation for research
- Synthesizes structured reports from retrieved context
"""

import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from aiohttp import web


class NeuroscienceResponseEngine:
    """Context-aware response engine that extracts and organizes knowledge
    from the messages/context it receives, producing realistic responses."""

    def generate_response(
        self, messages: List[Dict], tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        all_text = " ".join(
            m.get("content", "") or ""
            for m in messages
            if isinstance(m.get("content"), str)
        )
        lower = all_text.lower()
        last_user = self._get_last_user_message(messages)
        last_user_lower = last_user.lower()

        # Get the last system message (for detecting context type)
        last_system = ""
        for m in reversed(messages):
            if m.get("role") == "system":
                last_system = (m.get("content") or "").lower()
                break

        # Detect if we're inside goal_driven_loop (system prompt has OBJECTIVE / SUB-TASK)
        in_goal_loop = "=== objective ===" in last_system or "current sub-task" in last_system
        # Detect if this is a planning-specific request (user message is ABOUT planning)
        is_planning_request = (
            ("break this request" in last_user_lower or "actionable steps" in last_user_lower
             or "minimal number of clear" in last_user_lower)
            and not in_goal_loop
        )

        # 1. Intent classification (only when explicitly asked)
        if "classify" in last_user_lower and ("casual" in last_user_lower or "task" in last_user_lower):
            return {"content": "TASK", "tool_calls": None}
        # Also check system prompt for classification requests
        if "classify" in last_system and ("casual" in last_system or "task" in last_system):
            return {"content": "TASK", "tool_calls": None}

        # 2. Simple/Complex classification
        if "simple" in lower and "complex" in lower and "exactly one word" in lower:
            return {"content": "SIMPLE", "tool_calls": None}

        # 3. Tool result messages — generate completion with [FINAL_ANSWER_COMPLETE]
        if any(m.get("role") == "tool" for m in messages):
            return self._handle_tool_result(messages, all_text)

        # 4. Validation (YES/NO)
        if "has the expected outcome" in last_system or "fully achieved" in last_system:
            return {"content": "YES", "tool_calls": None}
        if "validation" in last_user_lower and len(last_user) < 200:
            return {"content": "YES", "tool_calls": None}

        # 5. TOOL CALLS — highest priority when inside goal loop with tools
        if tools and in_goal_loop:
            return self._generate_tool_call(messages, tools)

        # 6. Task decomposition / planning — only for genuine planning requests
        if is_planning_request or (
            "task decomposition" in last_user_lower
            and "break" in last_user_lower
        ):
            return {"content": self._generate_plan(last_user), "tool_calls": None}
        # Fallback: detect planning from system prompt pattern
        if (
            not in_goal_loop
            and ("task decomposition" in last_system or "break this request" in last_system)
        ):
            return {"content": self._generate_plan(last_user), "tool_calls": None}

        # 7. Concept extraction (JSON array)
        if "abstract concept" in lower and "json" in lower:
            return {"content": self._extract_concepts(all_text), "tool_calls": None}

        # 8. Multi-query generation for research
        if "search queries" in lower and "json array" in lower:
            return {"content": self._generate_queries(last_user, all_text), "tool_calls": None}

        # 9. Report synthesis
        if ("synthesize" in lower or "professional report" in lower or "structured report" in lower) and len(all_text) > 2000:
            return {"content": self._synthesize_report(last_user, all_text), "tool_calls": None}

        # 10. JSON topic extraction
        if "json" in lower and ("topic_name" in lower or "topic_description" in lower):
            return {
                "content": json.dumps({
                    "topic_name": "Neuroscience Research",
                    "topic_description": "Analysis of neural systems and cognitive processes"
                }),
                "tool_calls": None,
            }

        # 11. Tool calling (general case — outside goal loop)
        if tools and self._should_use_tools(last_user_lower, tools):
            return self._generate_tool_call(messages, tools)

        # 12. Default: contextual neuroscience response with completion marker
        return {"content": self._contextual_response(last_user, all_text), "tool_calls": None}

    def _get_last_user_message(self, messages: List[Dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
        return ""

    def _generate_plan(self, query: str) -> str:
        return json.dumps({
            "steps": [
                {
                    "description": f"Research: {query[:100]}",
                    "dependencies": [],
                    "suggested_tool": "GetContext",
                    "expected_outcome": "Retrieved relevant neuroscience context from knowledge base",
                },
                {
                    "description": "Synthesize findings into structured response",
                    "dependencies": ["step_1"],
                    "suggested_tool": "",
                    "expected_outcome": "Task completed with neuroscience-grounded analysis",
                },
            ]
        })

    def _should_use_tools(self, message: str, tools: List[Dict]) -> bool:
        action_words = [
            "research", "analyze", "find", "search", "retrieve",
            "context", "knowledge", "investigate", "report",
        ]
        return any(w in message for w in action_words) and len(tools) > 0

    def _generate_tool_call(self, messages: List[Dict], tools: List[Dict]) -> Dict[str, Any]:
        last_msg = self._get_last_user_message(messages)

        # Extract the actual query from the message (strip framework boilerplate)
        query = last_msg
        if "USER REQUEST:" in last_msg:
            query = last_msg.split("USER REQUEST:")[-1].strip()
        elif "Research:" in last_msg:
            query = last_msg.split("Research:")[-1].strip()
        query = query.replace("[No additional external context]", "").strip()
        if not query:
            query = last_msg[:200]

        # Priority order for tool selection: GetContext > ConductResearch > others (skip FindTool)
        best_tool = None
        for preferred in ["GetContext", "ConductResearch"]:
            for t in tools:
                fn = t.get("function", {})
                if fn.get("name") == preferred:
                    best_tool = t
                    break
            if best_tool:
                break

        if not best_tool:
            # Pick first non-FindTool tool
            for t in tools:
                fn = t.get("function", {})
                if fn.get("name", "").lower() != "findtool":
                    best_tool = t
                    break
            if not best_tool:
                best_tool = tools[0]

        fn = best_tool["function"]
        fn_name = fn["name"]
        params = fn.get("parameters", {})
        required = params.get("required", [])

        args = {}
        for pname in required:
            prop = params.get("properties", {}).get(pname, {})
            ptype = prop.get("type", "string")
            if ptype == "string":
                args[pname] = query[:300]
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
                    "function": {"name": fn_name, "arguments": json.dumps(args)},
                }
            ],
        }

    def _handle_tool_result(self, messages: List[Dict], all_text: str) -> Dict[str, Any]:
        """After tool execution, generate response based on retrieved context."""
        # Collect tool results
        tool_contents = []
        for m in messages:
            if m.get("role") == "tool":
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 50:
                    tool_contents.append(content)

        if tool_contents:
            combined = "\n".join(tool_contents)
            # Extract key neuroscience terms from the context
            neuro_terms = self._extract_neuro_terms(combined)
            summary = self._summarize_context(combined, neuro_terms)
            return {
                "content": f"{summary}\n\nTask completed successfully. [FINAL_ANSWER_COMPLETE]",
                "tool_calls": None,
            }

        return {
            "content": "The research has been completed. The knowledge base context has been analyzed. [FINAL_ANSWER_COMPLETE]",
            "tool_calls": None,
        }

    def _extract_neuro_terms(self, text: str) -> List[str]:
        """Extract neuroscience-related terms from text."""
        neuro_keywords = [
            "neuron", "synapse", "axon", "dendrite", "cortex", "hippocampus",
            "amygdala", "thalamus", "cerebellum", "prefrontal", "temporal",
            "dopamine", "serotonin", "glutamate", "GABA", "acetylcholine",
            "action potential", "membrane potential", "ion channel",
            "neurotransmitter", "receptor", "plasticity", "long-term potentiation",
            "myelination", "glial", "astrocyte", "oligodendrocyte",
            "basal ganglia", "brainstem", "spinal cord", "neural circuit",
            "synaptic transmission", "vesicle", "endocytosis", "exocytosis",
            "resting potential", "depolarization", "repolarization",
            "excitatory", "inhibitory", "postsynaptic", "presynaptic",
            "cortical", "subcortical", "limbic", "neocortex",
            "sensory", "motor", "autonomic", "somatic",
            "cognitive", "memory", "learning", "perception",
        ]
        found = []
        text_lower = text.lower()
        for term in neuro_keywords:
            if term in text_lower:
                found.append(term)
        return found[:20]

    def _summarize_context(self, context: str, neuro_terms: List[str]) -> str:
        """Generate a summary that references the actual context content."""
        # Extract some actual sentences from the context
        sentences = re.split(r'[.!?]\s+', context)
        relevant = [s.strip() for s in sentences if len(s.strip()) > 40][:8]

        parts = []
        if neuro_terms:
            parts.append(
                f"Based on the retrieved knowledge base context, the following neuroscience concepts are relevant: "
                f"{', '.join(neuro_terms[:10])}."
            )

        if relevant:
            parts.append("\nKey findings from the indexed literature:\n")
            for i, sent in enumerate(relevant[:5], 1):
                clean = sent.strip()[:200]
                parts.append(f"  {i}. {clean}")

        if not parts:
            parts.append("The knowledge base context has been analyzed and the information has been processed.")

        return "\n".join(parts)

    def _extract_concepts(self, all_text: str) -> str:
        """Generate concept extraction JSON from context."""
        neuro_terms = self._extract_neuro_terms(all_text)
        concepts = []
        for i, term in enumerate(neuro_terms[:5]):
            concepts.append({
                "name": term.title().replace(" ", "_"),
                "description": f"Neural concept related to {term} as found in the indexed neuroscience literature",
                "type": "neuroscience_concept",
                "score": round(0.95 - i * 0.05, 2),
            })
        if not concepts:
            concepts = [{
                "name": "Neural_Processing",
                "description": "General neural information processing concept from neuroscience literature",
                "type": "strategy",
                "score": 0.85,
            }]
        return json.dumps(concepts)

    def _generate_queries(self, last_user: str, all_text: str) -> str:
        """Generate diverse search queries for research."""
        base = last_user[:100] if last_user else "neuroscience"
        queries = [
            f"neuroscience {base}",
            f"neural circuits and {base}",
            f"synaptic mechanisms in {base}",
            f"cognitive neuroscience perspective on {base}",
            f"brain regions involved in {base}",
        ]
        return json.dumps(queries)

    def _synthesize_report(self, topic: str, all_text: str) -> str:
        """Synthesize a structured report from retrieved context."""
        neuro_terms = self._extract_neuro_terms(all_text)

        # Extract actual content snippets from the context
        sentences = re.split(r'[.!?]\s+', all_text)
        relevant_sentences = []
        for s in sentences:
            s = s.strip()
            if len(s) > 50 and any(t in s.lower() for t in ["neuron", "synap", "cortex", "brain",
                                                               "neural", "membrane", "potential",
                                                               "transmit", "receptor", "axon",
                                                               "dendrit", "hippocamp", "consolidat",
                                                               "memory", "learning", "cognit"]):
                relevant_sentences.append(s)

        report_sections = []
        report_sections.append(f"# Cross-Domain Analysis Report: {topic[:100]}\n")
        report_sections.append("## Executive Summary\n")
        report_sections.append(
            f"This report presents a cross-domain analysis drawing from neuroscience literature "
            f"indexed in the knowledge base. The analysis identified {len(neuro_terms)} relevant "
            f"neuroscience concepts that map to the computational architecture under study.\n"
        )

        if neuro_terms:
            report_sections.append("## Key Neuroscience Concepts Identified\n")
            for term in neuro_terms[:12]:
                report_sections.append(f"- **{term.title()}**: Referenced in indexed literature")

        if relevant_sentences:
            report_sections.append("\n## Evidence from Indexed Literature\n")
            for i, sent in enumerate(relevant_sentences[:10], 1):
                clean = sent[:250].strip()
                report_sections.append(f"{i}. {clean}")

        report_sections.append("\n## Cross-Domain Mapping\n")
        report_sections.append(
            "The knowledge consolidator in NeuralCore mirrors the biological process of memory "
            "consolidation observed in the hippocampus. The LambdaMART reranking model acts as "
            "a form of synaptic weighting, where frequently accessed and relevant knowledge items "
            "receive stronger 'synaptic connections' (higher relevance scores). This is analogous "
            "to long-term potentiation (LTP) in biological neural networks.\n"
        )
        report_sections.append(
            "The context manager's short-term memory buffer parallels the working memory system "
            "in the prefrontal cortex, while the persistent KnowledgeBase represents long-term "
            "declarative memory storage similar to hippocampal-cortical memory consolidation.\n"
        )

        report_sections.append("## Conclusions\n")
        report_sections.append(
            "The cross-domain RAG pipeline successfully retrieves and integrates neuroscience "
            "concepts from the indexed PDF literature, demonstrating that domain-specific knowledge "
            "can be leveraged for computational architecture analysis. The Learn-to-Rank model "
            "adapts its scoring features based on the neuroscience domain vocabulary, providing "
            "evidence of cross-domain knowledge transfer.\n"
        )

        return "\n".join(report_sections)

    def _contextual_response(self, last_user: str, all_text: str) -> str:
        """Generate a contextual response, incorporating any available neuroscience context."""
        neuro_terms = self._extract_neuro_terms(all_text)

        if neuro_terms:
            terms_str = ", ".join(neuro_terms[:6])
            response = (
                f"Based on the neuroscience knowledge available, I can address your query "
                f"regarding: {last_user[:120]}. The relevant neuroscience concepts include "
                f"{terms_str}. "
            )

            # Add domain-specific content based on detected terms
            if "synapse" in neuro_terms or "synaptic" in all_text.lower():
                response += (
                    "Synaptic transmission involves the release of neurotransmitters from "
                    "presynaptic vesicles into the synaptic cleft, where they bind to "
                    "postsynaptic receptors. This process is fundamental to neural communication. "
                )
            if "hippocampus" in neuro_terms or "memory" in neuro_terms:
                response += (
                    "The hippocampus plays a critical role in memory consolidation, transforming "
                    "short-term memories into long-term storage through repeated reactivation "
                    "of neural circuits during sleep and rest periods. "
                )
            if "action potential" in neuro_terms or "membrane" in all_text.lower():
                response += (
                    "Action potentials propagate along axons through voltage-gated ion channels, "
                    "with the all-or-none principle ensuring reliable signal transmission across "
                    "neural networks. "
                )
            if "plasticity" in neuro_terms or "learning" in neuro_terms:
                response += (
                    "Neural plasticity underlies learning and adaptation, with Hebbian learning "
                    "('neurons that fire together wire together') being a foundational principle "
                    "of synaptic modification. "
                )
            if "cortex" in neuro_terms or "prefrontal" in neuro_terms:
                response += (
                    "The prefrontal cortex is essential for executive functions including "
                    "working memory, decision-making, and cognitive control. Its layered "
                    "architecture enables hierarchical processing of information. "
                )

            response += "Task complete. [FINAL_ANSWER_COMPLETE]"
            return response

        return (
            f"Processing your request: {last_user[:120]}. "
            "The analysis is based on the available knowledge base context. "
            "Task complete. [FINAL_ANSWER_COMPLETE]"
        )


class AdvancedMockLLMServer:
    """OpenAI-compatible mock LLM server with neuroscience-aware response engine."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9111):
        self.host = host
        self.port = port
        self.engine = NeuroscienceResponseEngine()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self.request_log: List[Dict[str, Any]] = []
        self.request_count = 0

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
        print(f"[MOCK SERVER] Running at {self.base_url}")

    async def stop(self):
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        print(f"[MOCK SERVER] Stopped. Served {self.request_count} requests.")

    async def _handle_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [{"id": "mock-model", "object": "model", "owned_by": "lab"}],
        })

    async def _handle_embeddings(self, request: web.Request) -> web.Response:
        import numpy as np
        body = await request.json()
        dim = 384
        inp = body.get("input", "")
        if isinstance(inp, list):
            inp = " ".join(str(x) for x in inp)
        seed = hash(str(inp)) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(float)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return web.json_response({
            "object": "list",
            "data": [{"object": "embedding", "embedding": vec.tolist(), "index": 0}],
            "model": body.get("model", "mock-embed"),
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        })

    async def _handle_chat(self, request: web.Request) -> Union[web.Response, web.StreamResponse]:
        body = await request.json()
        self.request_count += 1
        self.request_log.append({
            "timestamp": time.time(),
            "model": body.get("model"),
            "stream": body.get("stream", False),
            "message_count": len(body.get("messages", [])),
            "has_tools": bool(body.get("tools")),
            "request_num": self.request_count,
        })

        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        model = body.get("model", "mock-model")

        result = self.engine.generate_response(messages, tools)
        content = result.get("content")
        tool_calls = result.get("tool_calls")

        try:
            if stream:
                return await self._stream_response(request, content, tool_calls, model)
            else:
                return self._non_stream_response(content, tool_calls, model)
        except (ConnectionResetError, ConnectionError, Exception) as e:
            if "closing transport" in str(e).lower() or "reset" in str(e).lower():
                return web.Response(status=499, text="client disconnected")
            raise

    def _non_stream_response(self, content, tool_calls, model):
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = "tool_calls" if tool_calls else "stop"
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 50, "completion_tokens": max(len(content or "") // 4, 10), "total_tokens": 60},
        })

    async def _stream_response(self, request, content, tool_calls, model):
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        await resp.prepare(request)
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        async def _write_chunk(data):
            await resp.write(f"data: {json.dumps(data)}\n\n".encode())

        if tool_calls:
            for tc_idx, tc in enumerate(tool_calls):
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                await _write_chunk({
                    "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": [
                        {"index": tc_idx, "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                         "type": "function", "function": {"name": fn.get("name", ""), "arguments": ""}}
                    ]}, "finish_reason": None}],
                })
                await asyncio.sleep(0.005)
                chunk_size = max(10, len(args_str) // 3)
                for i in range(0, len(args_str), chunk_size):
                    await _write_chunk({
                        "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": tc_idx, "function": {"arguments": args_str[i:i + chunk_size]}}
                        ]}, "finish_reason": None}],
                    })
                    await asyncio.sleep(0.005)
            await _write_chunk({
                "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            })
        elif content:
            words = content.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else f" {word}"
                delta = {"content": token}
                if i == 0:
                    delta["role"] = "assistant"
                await _write_chunk({
                    "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                })
                await asyncio.sleep(0.001)
            await _write_chunk({
                "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
        else:
            await _write_chunk({
                "id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            })

        await resp.write(b"data: [DONE]\n\n")
        return resp


if __name__ == "__main__":
    async def main():
        server = AdvancedMockLLMServer(port=9111)
        await server.start()
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            await server.stop()
    asyncio.run(main())
