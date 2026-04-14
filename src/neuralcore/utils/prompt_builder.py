from datetime import datetime
from typing import List, Dict

from neuralcore.utils.os_info import get_os_info
from neuralcore.utils.logger import Logger

logger = Logger.get_logger()


class PromptBuilder:
    """
    Centralized, reusable prompt builder for NeuralCore.
    Must remain completely abstract and free of any client-specific business logic.
    """

    user_system = get_os_info()
    current_time = datetime.now().isoformat()

    @staticmethod
    def shell_helper(user_input: str) -> str:
        """Generates a shell command prompt tailored to the actual distro."""
        return f"""
        Generate a SINGLE, non-interactive shell command for this system: **{PromptBuilder.user_system}**

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
        Time: {PromptBuilder.current_time}
        OS: {PromptBuilder.user_system}
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

    # ====================== NEW PROMPTS FROM AgentExecutors ======================

    @staticmethod
    def classify_intent(query: str) -> str:
        """Fast casual vs task intent classification."""
        return f"""Classify this user message as either CASUAL or TASK.

        CASUAL = greeting, small talk, "how are you", joke, opinion, thank you, chit-chat, storytelling, emotional support, philosophy, roleplay, simple general-knowledge questions.
        TASK   = anything that might need tools, search, calculation, research, file/code work, actions, multi-step goal, current events, data lookup, etc.

        User message: {query}

        Answer with **exactly one word**: CASUAL or TASK"""

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
        """Structured task decomposition prompt."""
        return f"""You are a task decomposition expert. Break the following user request into the minimal number of clear, sequential sub-tasks.

        USER REQUEST: {original_query}

        Rules (strict):
        - Each sub-task must be atomic and actionable.
        - Include explicit dependencies (previous step indices, 0-based).
        - If a step needs a tool that may not be loaded, note "may_require_FindTool".
        - Output ONLY valid JSON, nothing else:

        {{
        "steps": [
            {{
            "description": "exact sub-task description",
            "dependencies": [list of previous step indices or empty list],
            "suggested_tool_category": "optional short hint or empty string"
            }},
            ...
        ]
        }}

        Example for "open file A and reference it to analyze file B":
        {{
        "steps": [
            {{ "description": "Load and read the full content of file A", "dependencies": [], "suggested_tool_category": "file_read" }},
            {{ "description": "Analyze file B while referencing the loaded content from file A", "dependencies": [0], "suggested_tool_category": "file_analyze" }}
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
    ) -> str:
        """State-aware prompt for executing one specific sub-task in multi-step flow."""
        return f"""You are executing **ONE SPECIFIC SUB-TASK ONLY**. Ignore everything else.

        ORIGINAL USER REQUEST (background only): {original_query}

        ALREADY COMPLETED STEPS (do NOT repeat any of these actions or tools):
        {completed_context}

        TOOLS ALREADY USED IN THIS SESSION (NEVER call these again on any future turn):
        {used_tools_str}

        REMAINING STEPS (after you finish the current one):
        {remaining_context}

        CURRENT SUB-TASK ({current_index + 1}/{total_tasks}): {task_desc}

        STRICT EXECUTION PROTOCOL FOR THIS TURN ONLY:
        1. Focus EXCLUSIVELY on the CURRENT SUB-TASK above.
        2. If you need information from a previously read file, use the tool_results / KB context that is already provided — do NOT re-call any tool listed in "TOOLS ALREADY USED".
        3. If the required tool is missing, call FindTool first. After FindTool succeeds, call the newly loaded tool.
        4. After the tool for this sub-task succeeds and you have the result, you MUST immediately output EXACTLY this marker and nothing else:
        {marker}
        Do not add any commentary, do not say you are ready for the next step, do not continue thinking, do not summarize.

        You are now on turn {loop_count}. Stay on protocol. No deviations."""

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
        - When a sub-task is complete, output exactly: [FINAL_ANSWER_COMPLETE]
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
        return """You are an ACTION-ORIENTED AGENT with direct tool access.

        CORE RULES (never violate):
        - If the user request involves any concrete action (add, create, write, modify, execute, implement, read, analyze with tools, etc.) → CALL THE APPROPRIATE TOOL immediately.
        - NEVER respond with a summary, report, or explanation when a tool can fulfill the request.
        - You have access to write_file, read_file, execute commands, FindTool, and many others. Use them directly.
        - If the exact tool you need is not currently loaded, call FindTool first.
        - After a tool succeeds and the current request is complete, output exactly: [FINAL_ANSWER_COMPLETE]
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
    def context_summary_instruction() -> str:
        """Instruction for how to use the context summary block."""
        return "→ Use this context to stay aware. Never repeat the summary."

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
        objective_text: str, compact_history: str = "", current_subtask: str = ""
    ) -> str:
        """Ultra-compact context for long-running agentic loops.
        Used by the new provide_context_agentic_light() path."""
        parts = [f"=== OBJECTIVE ===\n{objective_text}"]
        if current_subtask:
            parts.append(f"=== CURRENT SUB-TASK ===\n{current_subtask}")
        if compact_history:
            parts.append(f"=== COMPACT HISTORY ===\n{compact_history}")
        parts.append(
            "=== END CONTEXT ===\nStay focused. Use tools. Output [FINAL_ANSWER_COMPLETE] when done."
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
            "[FINAL_ANSWER_COMPLETE]",
            "AUTONOMOUS CONTINUATION",
        ]
