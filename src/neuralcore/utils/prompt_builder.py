from datetime import datetime
from typing import Any, Dict, List, Optional
from neuralcore.agents.task import Task
from neuralcore.actions.registry import registry
from neuralcore.utils.os_info import get_os_info
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class PromptBuilder:
    """
    Centralized, reusable prompt builder for NeuralCore.
    Must remain completely abstract and free of any client-specific business logic.
    """

    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

    @staticmethod
    def shell_helper(user_input: str) -> str:
        """Generates a shell command prompt tailored to the actual distro."""
        return f"""
Generate a SINGLE, non-interactive shell command for this system: **{get_os_info()}**

Rules:
- Use the correct package manager (pacman for Arch, apt for Debian/Ubuntu, dnf for Fedora, zypper for openSUSE, etc.).
- Never explain, never wrap in code blocks, never add extra text.
- If the command needs sudo, include it.
- Always use flags to make it non-interactive (-y, --noconfirm, --assume-yes, etc.).
- Do NOT ask for confirmation.

User wants to: {user_input}
""".strip()

    @staticmethod
    def analyzer_helper(command: str, output: str) -> str:
        """Analyzes shell command output."""
        return f"""
Analyze the output of the following command: {command}

Output:
{output}

Summarize errors, warnings, success status, and key information.
Only mention system details if relevant to the issue.

SYSTEM INFO (use only if needed):
Time: {datetime.now().isoformat()}
OS: {get_os_info()}
""".strip()

    @staticmethod
    def topics_helper(history: list) -> str:
        """Extracts topic from conversation history."""
        history_text = str(history)
        logger.debug(f"Topics helper: injected history: {history_text}")

        return f"""
Based on the following conversation history, please name a topic and provide a description in JSON format:

{history_text}

Response must be valid JSON with keys:
- "topic_name": string
- "topic_description": string
""".strip()

    @staticmethod
    def analyze_code(content: str) -> str:
        """Analyzes code and returns structured metadata in JSON."""
        return f"""
Analyze the following code and return structured metadata in JSON format with these keys:
- "functions": list of {{ "name": str, "description": str }}
- "classes": list of {{ "name": str, "purpose": str }}
- "purpose": brief overall summary

Code:
{content}
""".strip()

    @staticmethod
    def casual_chat_system_prompt() -> str:
        """Warm, friendly system prompt for pure casual chat mode."""

        return (
            f"You are a friendly, natural, and engaging conversational AI assistant.\n"
            f"Current time: {datetime.now().isoformat()}\n"
            f"System: {get_os_info()}\n\n"
            "Respond casually, warmly, and helpfully. "
            "Keep replies concise but engaging. Use the time/OS info only if relevant."
        )

    @staticmethod
    def default_agent_system_prompt() -> str:
        """Default system prompt for agentic / tool-using modes."""

        return (
            f"You are a helpful, accurate, and concise AI assistant.\n"
            f"Current time: {datetime.now().isoformat()}\n"
            f"System: {get_os_info()}"
        )

    @staticmethod
    def classify_intent(query: str) -> str:
        """Return a clean, one-shot classification prompt."""
        return f"""You are an intent classifier. Classify the FINAL user message as exactly one of:

CASUAL → greeting, small talk, chit-chat, joke, opinion, thanks, storytelling, emotional support, philosophy, roleplay, simple factual questions that need no tools.

TASK   → anything that could benefit from tools, code, files, search, calculation, research, multi-step planning, current events, data lookup, actions, or workflow execution.

User message to classify:
{query}

Answer with **exactly one word** on its own line: CASUAL or TASK
Do not explain. Do not add any other text."""

    @staticmethod
    def casual_system_prompt() -> str:
        """System prompt for casual chat mode."""
        return """You are a warm, friendly, engaging AI companion.
        Be natural, use contractions, show personality, be concise unless asked otherwise.
        Never mention tools, internal steps, thinking process, or knowledge base unless the user explicitly asks about them."""

    @staticmethod
    def is_multi_step_task(query: str) -> str:
        """Detects if a request requires multiple steps."""
        return f"""You are an expert at analyzing user requests for multi-step execution.

Determine if this request is:
- SIMPLE → one direct action (single tool call)
- COMPLEX → requires **multiple distinct steps**, planning, sequencing, or dependencies

Be strict. Examples:
- "list files in this dir"          → SIMPLE
- "read config.json"                → SIMPLE
- "read this PDF and analyze this code from that point of view" → COMPLEX
- "open file A and reference it to analyze file B" → COMPLEX

Request: {query}

Answer with **exactly one word**: SIMPLE or COMPLEX"""

    @staticmethod
    def task_decomposition(original_query: str) -> str:
        """Pragmatic task decomposition.
        Automatically fetches the FULL (unlimited) list of all tools directly from registry.list_all_tools().
        Planning is now 100% grounded in the real, complete tool registry.
        Combined with multi-step detection: always plans, then caller decides singular vs multi based on plan size.
        """

        # ── Fetch the complete tool list directly from registry (NO LIMIT) ──
        available_tools = registry.list_all_tools(
            limit=None,  # ← explicit: no limit
            include_hidden=False,
            as_llm_format=False,
            include_schema=False,
        )

        # ── Build clean, readable tools section (grouped by set) ──
        tools_section = ""
        if available_tools:
            sorted_tools = sorted(
                available_tools,
                key=lambda t: (t.get("set_name", ""), t.get("name", "")),
            )

            tools_list = []
            for tool in sorted_tools:
                name = tool.get("name", "unknown")
                set_name = tool.get("set_name", "UnknownSet")
                desc = (tool.get("description", "") or "").strip()[:160]
                if desc and not desc.endswith("."):
                    desc += "."
                tools_list.append(f"• {name} [{set_name}] → {desc}")

            tools_text = "\n".join(tools_list)
            tools_section = f"""
    CURRENTLY AVAILABLE TOOLS (full list — use these exact tools whenever possible):

    {tools_text}

    You MUST prefer these real tools. Only suggest a step that cannot be fulfilled by any of them if absolutely necessary.
    """
        else:
            tools_section = "\nNo tools are currently registered in the system."

        return f"""You are a pragmatic task decomposition expert.

USER REQUEST: {original_query}

{tools_section}

Break this request into the **minimal number of clear, actionable steps** required to complete the goal.

Core Rules (strict):
- **SINGULAR vs MULTI DECISION RULE**: 
- Output **EXACTLY 1 step** ONLY if the entire request can be completed with a **single tool call or single atomic action** with no real dependencies.
- Output **2 or more steps** whenever the request contains **sequential phases**, **different capabilities**, or **hard dependencies** (e.g. "read file → analyze content → write result", "search → process results → save", etc.).
- Common patterns that MUST be split: file reading + content analysis + file writing; data fetching + transformation + output; any request with "first... then..." or multiple distinct phases.

- Do NOT collapse genuinely sequential or multi-capability tasks into one step just to be minimal. Accuracy of phase separation is more important than having the smallest possible number of steps.
- Each step must correspond to what **one realistic tool execution** can achieve.
- For "suggested_tool", use the **exact tool name** from the list above when possible. Otherwise use a short category (e.g. "file_read", "code_analysis", "file_write").
- For "expected_outcome", keep it short and practical (e.g. "File successfully read", "Analysis completed", "Report written to disk").

Output ONLY valid JSON:

{{
"steps": [
    {{
    "description": "exact sub-task description",
    "dependencies": [list of previous step indices or empty list],
    "suggested_tool": "exact tool name if available, otherwise short category hint",
    "expected_outcome": "short, realistic success description"
    }}
]
}}

Return the plan:"""

    @staticmethod
    def sub_task_execution(
        original_query: str,
        task_desc: str,
        current_index: int,
        total_tasks: int,
        completed_context: str,
        used_tools_str: str,
        remaining_context: str,
        marker: str,
        loop_count: int,
        expected_outcome: str = "",
        previous_results_context: str = "",
    ) -> str:
        is_final_step = current_index == total_tasks - 1

        final_step_block = ""
        if is_final_step:
            final_step_block = """
            THIS IS THE FINAL SUB-TASK.
            The overall goal is complete as soon as this sub-task's expected_outcome is fully achieved.
            Focus only on completing this last step correctly.
            When the expected_outcome is met, output the marker."""

        expected_block = ""
        if expected_outcome and expected_outcome.strip():
            expected_block = f"""
            EXPECTED OUTCOME THAT MUST BE VERIFIED BEFORE EMITTING MARKER:
            {expected_outcome.strip()}

            After the tool result arrives, you MUST mentally verify:
            - Did the tool produce exactly what the expected_outcome describes?
            - Are arguments correct and the result meaningful?
            Only if YES → output exactly the marker below. Otherwise continue working on this sub-task."""

        previous_results_block = ""
        if previous_results_context and previous_results_context.strip():
            previous_results_block = f"""
            PREVIOUS RESULTS (FULL CONTENT — reuse this, DO NOT re-read files):
            {previous_results_context}
            """

        return f"""You are executing **ONE SPECIFIC SUB-TASK ONLY**. Ignore everything else.

ORIGINAL USER REQUEST (background only): {original_query}

ALREADY COMPLETED STEPS (short summary — do NOT repeat):
{completed_context}
{previous_results_block}

TOOLS ALREADY USED (avoid exact repeats unless necessary):
{used_tools_str}

TOOL RULE: If FindTool succeeded last turn, the tool is loaded — use it directly. Do not call FindTool again.

REMAINING STEPS:
{remaining_context}

CURRENT SUB-TASK ({current_index + 1}/{total_tasks}): {task_desc}
{expected_block}
{final_step_block}

STRICT PROTOCOL:
1. Focus only on the current sub-task.
2. Call the correct tool with accurate parameters.
3. After tool result: verify it matches the expected_outcome above.
4. When verification passes → output EXACTLY:
{marker}
Nothing else. No commentary.

Turn: {loop_count}"""

    @staticmethod
    def objective_reminder(reminder_body: str) -> str:
        """Centralized objective reminder template.
        Takes the dynamic state-built body and wraps it with the fixed header + critical instructions."""
        return f"""OBJECTIVE REMINDER:
{reminder_body}

CRITICAL INSTRUCTIONS:
- Stay focused on the current goal and sub-task.
- If the required tool for the current action is missing, FIRST use the FindTool tool to discover and load it.
- Only after the needed tool has been successfully loaded via FindTool should you call the actual tool.
- When a sub-task is complete, output exactly: 
- Use only verified information from tool_results when summarizing."""

    @staticmethod
    def final_synthesis(original_query: str) -> str:
        """Final answer synthesis prompt."""
        return f"""USER ORIGINAL REQUEST: {original_query}

Task is now complete (or max loops reached). 
Summarize using ONLY the verified results from state.tool_results. Be concise."""

    @staticmethod
    def findtool_refinement(original_user_query: str) -> str:
        """Refines user request for FindTool selection."""
        return f"""You are an expert tool-router.
Convert this user request into a SHORT, keyword-rich query (max 12 words) that will perfectly match the best tool in our registry.

USER REQUEST: {original_user_query}

Rules:
- Focus ONLY on the core capability needed (e.g. "web search weather", "read pdf file", "search codebase function").
- Use terms that appear in tool names/descriptions/tags.
- Do NOT include specific data values (e.g. do NOT put "Poland").
- Return ONLY the refined query as plain text, nothing else.

Refined query:"""

    @staticmethod
    def findtool_selection(
        refined_query: str, original_user_query: str, tool_list_str: str
    ) -> str:
        """Selects the best tool from the registry."""
        return f"""REFINED INTENT (use this for matching): {refined_query}

USER ORIGINAL REQUEST: {original_user_query}

AVAILABLE TOOLS:
{tool_list_str}

You MUST pick the SINGLE best tool that can fulfill the refined intent.
Return ONLY valid JSON (no extra text, no markdown):

{{
"tool_name": "exact_tool_name",
"parameters": {{ "key": "value", ... }},
"reason": "one short sentence"
}}

If no tool is clearly better than others, still pick the closest one."""

    @staticmethod
    def agentic_action_system_prefix() -> str:
        """Strong, always-present instruction that forces tool usage over explanation in agentic mode."""
        return f"""You are an ACTION-ORIENTED AGENT with direct tool access.

CORE RULES (never violate):
- If the user request involves any concrete action (add, create, write, modify, execute, implement, read, analyze with tools, etc.) → CALL THE APPROPRIATE TOOL immediately.
- NEVER respond with a summary, report, or explanation when a tool can fulfill the request.
- You have access to write_file, read_file, execute commands, FindTool, and many others. Use them directly.
- If the exact tool you need is not currently loaded, call FindTool first.
- After a tool succeeds and the current request is complete, output exactly: {PromptBuilder.FINAL_ANSWER_MARKER}
- Do not add commentary like "Based on previous analysis..." unless explicitly asked.

Stay in tool-execution mode until the goal is achieved."""

    @staticmethod
    def loaded_tools_summary(loaded_tool_names: List[str]) -> str:
        """Helper to show currently loaded tools."""
        if not loaded_tool_names:
            return "No tools are currently loaded. Call FindTool if you need any capability."
        return "Currently loaded tools:\n" + "\n".join(
            f"• {name}" for name in loaded_tool_names
        )

    @staticmethod
    def tool_expectations_helper(expected_outcome: str) -> str:
        """Generates the expectations block for the current sub-task.
        Completely abstract — only formats the success criteria."""
        if not expected_outcome or not expected_outcome.strip():
            return ""

        return f"""Expected outcome for current step: {expected_outcome.strip()}

When this outcome is achieved (e.g. file successfully modified, 
new tool added and working), you MUST output exactly the marker 
{PromptBuilder.FINAL_ANSWER_MARKER} and nothing else after it."""

    @staticmethod
    def context_summary_instruction() -> str:
        """Instruction for how to use the context summary block."""
        return "→ Use this context to stay aware."

    @staticmethod
    def recent_logs_section(log_lines: list) -> str:
        """Formats recent logs as a clean section."""
        if not log_lines:
            return ""
        return "=== RECENT LOGS ===\n" + "\n".join(log_lines) + "\n=== END LOGS ==="

    @staticmethod
    def objective_and_state_section(
        objective_text: str, loaded_tools_summary: str = ""
    ) -> str:
        """Combines objective reminder with optional loaded tools info."""
        parts = [f"=== OBJECTIVE & CURRENT STATE ===\n{objective_text}"]
        if loaded_tools_summary:
            parts.append(loaded_tools_summary)
        parts.append("=== END STATE ===")
        return "\n\n".join(parts)

    @staticmethod
    def lightweight_agentic_context(
        agent_state,
    ) -> str:
        """Ultra-compact context optimized for long-running lightweight agentic loops.

        Now receives the full AgentState object instead of individual extracted strings.
        Completely generic and reusable — no client-specific assumptions.
        """
        # Core identity & goal
        objective_text = agent_state.get_objective_reminder()

        # Current sub-task (safe lookup)
        current_subtask = ""
        if (
            agent_state.planned_tasks
            and isinstance(agent_state.current_task_index, int)
            and 0 <= agent_state.current_task_index < len(agent_state.planned_tasks)
        ):
            current_subtask = agent_state.planned_tasks[agent_state.current_task_index]

        # Currently loaded tools
        loaded_tools = ""
        if agent_state.loaded_tools:
            loaded_tools = PromptBuilder.loaded_tools_summary(agent_state.loaded_tools)

        # Recent tool activity (compact)
        compact_history = ""
        if hasattr(agent_state, "tool_results") and agent_state.tool_results:
            parts = []
            # Use last 10 tool results for compact history
            for entry in reversed(agent_state.tool_results[-10:]):
                tool_name = entry.get("name", "unknown")
                raw_result = str(entry.get("result", ""))

                if tool_name == "FindTool":
                    loaded = (
                        "loaded tools" in raw_result.lower()
                        or "success" in raw_result.lower()
                    )
                    preview = raw_result[:180].replace("\n", " ")
                    status = "SUCCESS" if loaded else "ATTEMPT"
                    parts.append(f"[FindTool {status}] → {preview}...")
                else:
                    preview = raw_result[:140].replace("\n", " ")
                    if len(raw_result) > 140:
                        preview += " [...]"
                    parts.append(f"[{tool_name}] → {preview}...")
            compact_history = "\n".join(parts)

        # Tool expectations for current step
        tool_expectations = ""
        if (
            agent_state.task_expected_outcomes
            and isinstance(agent_state.current_task_index, int)
            and 0
            <= agent_state.current_task_index
            < len(agent_state.task_expected_outcomes)
        ):
            expected = agent_state.task_expected_outcomes[
                agent_state.current_task_index
            ]
            tool_expectations = PromptBuilder.tool_expectations_helper(expected)

        # Build the context
        parts = [f"=== OBJECTIVE ===\n{objective_text}"]

        if current_subtask.strip():
            parts.append(f"=== CURRENT SUB-TASK ===\n{current_subtask}")

        if loaded_tools.strip():
            parts.append(f"=== CURRENTLY LOADED TOOLS ===\n{loaded_tools}")

        if compact_history.strip():
            parts.append(f"=== RECENT TOOL ACTIVITY (last 10) ===\n{compact_history}")

        if tool_expectations.strip():
            parts.append(f"=== CURRENT STEP EXPECTATIONS ===\n{tool_expectations}")

        parts.append(
            "=== STRICT EXECUTION RULES ===\n"
            "• If FindTool succeeded on the previous turn, the required tool is now loaded — DO NOT call FindTool again.\n"
            "• Use only the currently loaded tools to complete the CURRENT SUB-TASK.\n"
            f"• When the expected outcome for the current step is achieved, output exactly: {PromptBuilder.FINAL_ANSWER_MARKER}\n"
            "• Stay focused. No extra commentary."
        )

        return "\n\n".join(parts)

    @staticmethod
    def tool_call_history_section(call_history_text: str, last_three_text: str) -> str:
        """Formats tool history and recent outputs."""
        if not call_history_text and not last_three_text:
            return ""

        block = "=== TOOL CALL HISTORY & RECENT OUTPUTS ===\n"
        if call_history_text:
            block += (
                f"Recent tool calls (with parameters only):\n{call_history_text}\n\n"
            )
        if last_three_text:
            block += (
                f"Full output from the last 3 tool executions:\n{last_three_text}\n"
            )
        block += (
            "Use the call history to avoid repeating the same actions. "
            "Use the full outputs directly when they are relevant to the current goal."
        )
        return block

    @staticmethod
    def relevant_external_context_section(kb_text: str) -> str:
        """Formats KB / external context block."""
        if not kb_text:
            return ""
        return f"=== RELEVANT EXTERNAL CONTEXT ===\n{kb_text}\n=== END EXTERNAL CONTEXT ==="

    @staticmethod
    def knowledge_retrieval_embedding_query(user_query: str) -> str:
        """Query prefix used for embedding-based retrieval."""
        return f"Search query for relevant tool results and context: {user_query}"

    @staticmethod
    def knowledge_block_header(source_type: str, key: str, meta_str: str) -> str:
        """Formats the header for each knowledge block."""
        return f"[{source_type.upper()}] {key}\nMetadata: {meta_str}\n"

    @staticmethod
    def knowledge_block_separator() -> str:
        """Separator between knowledge blocks."""
        return "─" * 50 + "\n"

    @staticmethod
    def retrieve_relevant_knowledge_summary(
        returned_chars: int,
        tool_outcome_count: int,
        blocked_count: int,
        total_scored: int,
    ) -> str:
        """Optional debug-style summary (can be used in logs if needed)."""
        return (
            f"retrieve_relevant_knowledge FINISHED | "
            f"returned_chars={returned_chars} | "
            f"tool_outcomes_included={tool_outcome_count} | "
            f"blocked_contaminated={blocked_count} | "
            f"total_scored={total_scored}"
        )

    @staticmethod
    def context_summary_header() -> str:
        return "📋 LIVE CONTEXT SUMMARY"

    @staticmethod
    def context_summary_chat_header() -> str:
        return "📋 CHAT CONTEXT (light)"

    @staticmethod
    def context_summary_files_section(files: List[str]) -> str:
        return f"• Files checked: {', '.join(files) if files else '—'}"

    @staticmethod
    def context_summary_tools_section(tools: List[str]) -> str:
        return f"• Tools used: {', '.join(tools) if tools else '—'}"

    @staticmethod
    def context_summary_kb_section(kb_count: int) -> str:
        return f"• KB items: {kb_count}"

    @staticmethod
    def context_summary_token_section(current: int, max_tokens: int) -> str:
        warning = (
            "⚠️ Approaching context window limit..."
            if current > max_tokens * 0.8
            else ""
        )
        return f"• Estimated tokens: {current} / {max_tokens} {warning}"

    @staticmethod
    def context_summary_topic_section(topic_name: str) -> str:
        return f"• Active topic: {topic_name}"

    @staticmethod
    def context_summary_last_messages_header() -> str:
        return "📝 LAST MESSAGES"

    @staticmethod
    def context_summary_investigation_header() -> str:
        return "🧠 INVESTIGATION STATE"

    @staticmethod
    def context_summary_investigation_block(
        goal: str, pending: List[str], completed: List[str]
    ) -> str:
        return (
            f"• Goal: {goal or '—'}\n"
            f"• Pending: {', '.join(pending[:5]) if pending else '—'}\n"
            f"• Completed: {', '.join(completed[-5:]) if completed else '—'}"
        )

    @staticmethod
    def task_context_header(task_name: str) -> str:
        return f"🔹 TASK CONTEXT: {task_name.upper()}"

    @staticmethod
    def task_context_important_files_section(files: List[str]) -> str:
        return f"Important files: {', '.join(sorted(files))[:12]}"

    @staticmethod
    def task_context_key_results_section(results: List[Dict]) -> str:
        return "\n".join(f"• {r['title']}: {r['summary']}" for r in results[-10:])

    @staticmethod
    def task_context_findings_section(findings: List[str]) -> str:
        return "Key Findings:\n" + "\n".join(f"   - {f}" for f in findings[-8:])

    @staticmethod
    def task_context_hypotheses_section(hypotheses: List[str]) -> str:
        return "Hypotheses:\n" + "\n".join(f"   - {h}" for h in hypotheses[-5:])

    @staticmethod
    def latest_tool_block(
        tool_name: str, timestamp_str: str, size: int, content: str
    ) -> str:
        return (
            f"🔥 LATEST TOOL: {tool_name}\n"
            f"{timestamp_str}"
            f"Result size: {size:,} characters\n\n"
            f"{content}\n"
            f"{'─' * 90}\n"
        )

    @staticmethod
    def contamination_forbidden_phrases() -> List[str]:
        """Centralized list used by add_external_content and _retrieve_relevant_knowledge."""
        return [
            "You are a helpful Terminal AI agent",
            "CONTEXT WINDOW STATUS",
            "RELEVANT EXTERNAL CONTEXT",
            "CHAT CONTEXT (light)",
            f"{PromptBuilder.FINAL_ANSWER_MARKER}",
            "AUTONOMOUS CONTINUATION",
        ]

    # ====================== FINAL ANSWER & AGENTFLOW PROMPTS ======================

    @staticmethod
    def inject_final_answer_instruction(base_prompt: str) -> str:
        """Strong final answer instruction used across agent modes."""
        return f"""{base_prompt}

    CRITICAL FINAL ANSWER RULE:
    When you have **fully completed** the assigned task/micro-task and verified all required outputs, you MUST end your response with **exactly** this marker on its own line:

    {PromptBuilder.FINAL_ANSWER_MARKER}

    Do not add anything after the marker. Use it only when the goal is 100% achieved."""

    @staticmethod
    def deploy_chat_system_prompt(goal: str = "General assistance") -> str:
        """Main system prompt for the deploy/chat agent."""
        base = f"""You are a helpful Deploy Agent that executes commands immediately.

        RULES:
        - Use tool browser to load missing tools.
        - Keep responses short and natural after tool results.
        - Current goal: {goal}"""
        return PromptBuilder.inject_final_answer_instruction(base)

    @staticmethod
    def sub_agent_system_prompt(task_desc: str, assigned_tools: List[str] = []) -> str:
        """System prompt for sub-agents executing a single micro-task."""
        tools_hint = ""
        if assigned_tools:
            tools_list = ", ".join(assigned_tools[:15])
            if len(assigned_tools) > 15:
                tools_list += ", ..."
            tools_hint = f"\n\nAvailable tools: {tools_list}"

        base = f"""You are a precise sub-agent executing **ONE single micro-task only**.

        TASK: {task_desc}{tools_hint}

        CRITICAL RULES:
        - Complete ONLY this exact task.
        - If the task involves reading a file, use open_file_async or open_file_sync directly.
        - When you have finished the task, output a short summary and end with exactly: {PromptBuilder.FINAL_ANSWER_MARKER}"""
        return PromptBuilder.inject_final_answer_instruction(base)

    @staticmethod
    def plan_microtasks_prompt(task: str) -> str:
        """Orchestrator prompt for breaking a complex task into micro-tasks."""
        return f"""Break this task into 5-8 small focused micro-tasks.

TASK: {task}

Return ONLY valid JSON in this exact format:
{{
"microtasks": [
    {{
    "description": "Clear one-sentence description",
    "suggested_tools": ["tool1", "tool2"],
    "depends_on": null
    }},
    {{
    "description": "...",
    "suggested_tools": ["tool3"],
    "depends_on": "step_1"
    }}
]
}}

Note: Use "depends_on": null for tasks that can start immediately.
Use the first few words of a previous task's description as "depends_on" value if it depends on it."""

    @staticmethod
    def user_friendly_task_summary_prompt(
        task: str, goal: str = "", tool_results_str: str = ""
    ) -> str:
        """Natural summary prompt after a complex task completes."""
        return f"""You are a helpful Deploy Agent. The complex task has just finished.

Task: {task}
Goal: {goal or "General deployment assistance"}

What was actually done (tool results):
{tool_results_str or "No tool results recorded."}

Write a **friendly, concise, natural** message to the user (2–6 sentences max).
- Celebrate what was accomplished
- Mention any important outcomes or warnings
- End by saying we're back in normal chat mode and ask how else you can help

Tone: professional but warm and clear. No JSON. No technical jargon unless necessary.
"""

    @staticmethod
    def agent_objective_reminder(
        state,
    ) -> str:  # or accept individual fields if you prefer pure data
        """Rich, state-aware objective reminder (used by both main agent and sub-agents)."""
        if not state:
            return "Current goal: No goal set."

        parts = [f"Current goal: {state.goal or 'Complete the assigned task'}"]

        # Sub-task / progress awareness
        if state.planned_tasks and len(state.planned_tasks) > 1:
            current_idx = state.current_task_index + 1
            total = len(state.planned_tasks)
            parts.append(f"Progress: Sub-task {current_idx}/{total}")

        if state.current_task:
            parts.append(f"Current micro-task: {state.current_task[:150]}...")

        if state.tool_results:
            parts.append(f"Available tool results: {len(state.tool_results)}")

        if state.empty_loops > 0:
            parts.append(f"Empty loops counter: {state.empty_loops}")

        if state.phase:
            parts.append(f"Current phase: {state.phase}")

        # Critical tool usage rule (centralized)
        parts.append(
            "CRITICAL TOOL USAGE RULE:\n"
            "- If the tool you need is missing, FIRST call FindTool to discover and load it.\n"
            "- ONLY after FindTool has successfully loaded the tool should you call the actual tool."
        )

        # Termination marker (already in FINAL_ANSWER_MARKER constant)
        parts.append(
            f"\nWhen you have FULLY completed the current micro-task, "
            f"you MUST end your final response with exactly:\n{PromptBuilder.FINAL_ANSWER_MARKER}"
        )

        return "\n\n".join(parts)

    @staticmethod
    def sub_agent_objective_reminder(state) -> str:
        """Convenience alias — currently identical to objective_reminder but kept separate for future divergence."""
        return PromptBuilder.objective_reminder(state)

    @staticmethod
    def task_execution_summary_prompt(task: str, tool_results_str: str = "") -> str:
        """Exact relocation of the original _generate_deployment_summary prompt.
        Centralized, reusable, and free of any client-specific logic."""
        return f"""You are a helpful Deploy Agent. A complex background task has just finished.

Task: {task}

Key results from tools:
{tool_results_str or "No detailed tool output available."}

Write a friendly, concise summary (3-7 sentences) for the user.
- Mention what was accomplished
- Highlight any important outcomes or warnings
- Use natural language and light emojis if appropriate
- Keep it easy to read"""

    @staticmethod
    def abstract_concept_extraction(
        goal: str,
        relevant_items: List[Any],
        existing_concepts: Optional[Dict[str, Any]] = None,
        max_concepts: int = 7,
    ) -> str:
        """High-precision prompt for extracting grounded, high-dimensional abstractions."""

        items_text = "\n\n".join(
            f"ITEM {i + 1} [source: {getattr(item, 'source_type', 'unknown')} | category: {getattr(item, 'category_path', 'N/A')}]\n"
            f"{getattr(item, 'content', '')[:1500]}"
            for i, item in enumerate(relevant_items[:12])
        )

        existing_text = ""
        if existing_concepts and len(existing_concepts) > 0:
            existing_text = (
                "\n\nEXISTING CONCEPTS (you MUST prefer refining, extending, or connecting to these):\n"
                + "\n".join(
                    f"- {name}: {concept.get('description', '')[:260]}"
                    for name, concept in list(existing_concepts.items())[:8]
                )
            )

        return f"""You are a precise systems thinker and abstraction specialist.

Your task is to extract **{max_concepts} high-quality, higher-dimensional abstract concepts** that reveal deep connections between the provided artifacts.

CRITICAL RULES (strictly enforced):
- ONLY extract a concept if there is **clear, direct evidence** in the items.
- Never invent mechanisms, classes, or processes that are not explicitly present.
- Prefer **refining or extending** existing concepts over creating new ones.
- Focus on **cross-domain intellectual bridges** (e.g. "LambdaMART reranking as computational analogue of attentional modulation").
- Each concept must feel like a genuine, non-obvious insight — not generic description.
- Output **ONLY** a valid JSON array. No explanations, no markdown, no extra text.

Goal: {goal}



Findings:
{existing_text}

RELEVANT ARTIFACTS:
{items_text}

Output format (JSON array only):
[
{{
    "name": "Short, precise, evocative name",
    "description": "1-2 sentence intellectually deep mapping",
    "type": "mechanism | strategy | principle | analogy | architecture",
    "score": 0.0-1.0,
    "links_to": ["existing_concept_name"] or []
}}
]
"""

    @staticmethod
    def context_summary_recent_tools_header() -> str:
        return "Recent Tool Outcomes (last 4):"

    @staticmethod
    def step_validation_prompt(
        current_task: "Task",
        last_result_str: str,
        total_tasks: int,
        current_idx: int,
    ) -> str:
        """Strict LLM-based step outcome validation prompt."""
        return f"""You are a strict validation agent for multi-step task execution.

CURRENT SUB-TASK ({current_idx + 1}/{total_tasks}):
{current_task.description}

EXPECTED OUTCOME THAT MUST BE VERIFIED:
{current_task.expected_outcome or "Step completed successfully"}

MOST RECENT TOOL RESULT:
{last_result_str[:2500] or "No tool results available yet."}

QUESTION:
Has the expected outcome been FULLY achieved based on the tool result above?
Be conservative. Only answer YES if the outcome is clearly and completely met.

Answer with **exactly one word** on its own line:
YES
or
NO

No explanations. No extra text."""

    @staticmethod
    def sub_task_execution_system_prompt() -> str:
        """System prompt used when building context for sub-task execution."""
        return """You are executing ONE specific sub-task in a larger plan.
Focus only on the current sub-task description provided.
Use tools when needed.
When the current sub-task is complete and the expected outcome is achieved, output exactly the marker."""

    @staticmethod
    def final_synthesis_system_prompt() -> str:
        """Clean system prompt for final answer synthesis."""
        return """FINAL ANSWER MODE
Provide a clear, complete, and concise summary of what was accomplished in the entire task.
Use only verified information from tool results.
Be natural and professional."""

    @staticmethod
    def validation_system_prompt() -> str:
        """System prompt specifically for step validation."""
        return """You are a strict validation agent.
Evaluate whether the expected outcome for the current sub-task has been achieved.
Be precise, conservative, and objective.
Use ONLY the information provided in the query.
Answer with exactly one word: YES or NO."""

    @staticmethod
    def analysis_multi_query_generation(analysis_query: str) -> str:
        """Generates 4-6 diverse and effective search queries for any given topic."""
        return f"""
You are an expert research strategist. 

Given the following topic or analysis goal, generate **4 to 6 diverse, high-quality search queries** 
that will help retrieve the most relevant and useful information from a knowledge base.

**Rules:**
- Make the queries cover different angles and aspects of the topic.
- Include a mix of broad exploratory queries and very specific, targeted ones.
- Use precise terminology that is likely to appear in documents, code, tool outputs, or reports.
- Vary the scope: conceptual, practical, comparative, problem-oriented, etc.
- Return **ONLY** a valid JSON array of strings. No explanations, no extra text, no markdown.

**Topic / Analysis Goal:**
{analysis_query}

**Output (JSON array only):**
["query one here", "query two here", "query three here", ...]
""".strip()

    @staticmethod
    def analysis_report_synthesis(original_query: str, combined_research: str) -> str:
        """Synthesizes a detailed, professional, and comprehensive analysis report."""
        return f"""
You are a senior research scientist and technical report writer with 20+ years of experience producing high-quality analysis for technical and scientific audiences.

Your task is to produce a **comprehensive, well-elaborated, professional research report** (target length: 1200–2000 words) based **exclusively** on the tool outcomes provided below. Do not add external knowledge.

**Original Research Query:**
{original_query}

**Gathered Research Data (from multiple tool executions):**
{combined_research}

**Required Report Structure** (use clear markdown headings):

1. **Executive Summary** (200-350 words)
- 5–7 sentences that capture the core question, key findings, and main conclusions.
- Write this section last.

2. **Research Context & Scope**
- Restate the original query in your own words.
- Briefly describe how the research was conducted (multiple targeted searches via GetContext and/or web indexing).
- Note the volume and nature of the evidence gathered.

3. **Key Findings** (detailed, 8–15 bullet points or short paragraphs)
- Each finding must include:
    - A clear statement
    - Specific evidence or quote from the research (with citation like [Local KB – "sub-query about X"] or [Web – search result Y])
    - Why it matters

4. **In-Depth Analysis & Synthesis** (600–1200 words)
- Go beyond listing findings.
- Identify patterns, trends, contradictions, and consensus across different tool outputs.
- Discuss implications for the original query.
- Connect technical details to broader context or consequences.
- Use sub-headings if helpful (e.g., ### Technical Implications, ### Strategic Considerations).

5. **Limitations, Gaps & Uncertainties**
- What is missing, unclear, or weakly supported in the gathered data?
- Any contradictions between sources?
- What additional research would be valuable?

6. **Conclusions & Actionable Recommendations**
- Clear, numbered recommendations (short-term and long-term).
- Prioritize based on strength of evidence.
- Suggest concrete next steps or experiments.

**Writing Guidelines:**
- Be objective and evidence-based. Never hallucinate or invent details.
- Cite specific tool results frequently using inline citations.
- Use professional, precise language. Avoid filler or overly verbose prose.
- Use markdown formatting: headings, bullet points, numbered lists, bold/italic for emphasis, and tables if comparing multiple items.
- Do **not** make the report short for the sake of brevity. Expand each section fully using all available relevant evidence.
- If the research data is limited, state that clearly instead of padding.

End the report with:
**Report generated on [current date/time] using ConductResearch tool.**

Produce the full report now.
""".strip()
