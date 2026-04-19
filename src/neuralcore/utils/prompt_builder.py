from datetime import datetime
from typing import Any, Dict, List, Optional

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
    FINAL_ANSWER_MARKER = "[FINAL_ANSWER_COMPLETE]"

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
        return f"""You are a task decomposition expert. Break the following user request into the minimal number of clear, sequential sub-tasks.

        USER REQUEST: {original_query}

        Rules (strict):
        - Each sub-task must be atomic and actionable.
        - Include explicit dependencies (previous step indices, 0-based).
        - For each step, define what SUCCESS looks like (expected_outcome).
        - If a step needs a tool that may not be loaded, note "may_require_FindTool".
        - Output ONLY valid JSON:

        {{
        "steps": [
            {{
            "description": "exact sub-task description",
            "dependencies": [list of previous step indices or empty list],
            "suggested_tool_category": "optional short hint",
            "expected_outcome": "short description of what must be true when this step is done (e.g. 'file terminal_set.py now contains new async execute_command tool')"
            }},
            ...
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

        === IMPORTANT: TOOL AVAILABILITY ===
        - If FindTool was called on the previous turn, the required tool(s) have now been successfully loaded.
        - DO NOT call FindTool again in this turn.
        - Use the newly loaded tool(s) directly to complete the CURRENT SUB-TASK.
        - You have access to write_file, read_file, search_code, etc. when they appear in the loaded tools.

        REMAINING STEPS (after you finish the current one):
        {remaining_context}

        CURRENT SUB-TASK ({current_index + 1}/{total_tasks}): {task_desc}

        STRICT EXECUTION PROTOCOL FOR THIS TURN ONLY:
        1. Focus EXCLUSIVELY on the CURRENT SUB-TASK above.
        2. If the required tool for this sub-task is already loaded, call it directly with correct parameters.
        3. After the tool for this sub-task succeeds and you have the result, you MUST immediately output EXACTLY this marker and nothing else:
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
        objective_text: str,
        compact_history: str = "",
        current_subtask: str = "",
        loaded_tools: str = "",
        tool_expectations: str = "",
    ) -> str:
        """Ultra-compact context optimized for long-running lightweight agentic loops."""
        parts = [f"=== OBJECTIVE ===\n{objective_text}"]

        if current_subtask:
            parts.append(f"=== CURRENT SUB-TASK ===\n{current_subtask}")

        if loaded_tools:
            parts.append(f"=== CURRENTLY LOADED TOOLS ===\n{loaded_tools}")

        if compact_history:
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
        hypotheses: List[str],
        findings: List[str],
        relevant_items: List[Any],
        existing_concepts: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Strongly guided prompt that forces use of hypotheses and findings
        while encouraging refinement of existing concepts."""

        # Format hypotheses and findings clearly
        hyp_text = "\n".join(f"- {h}" for h in hypotheses) if hypotheses else "None provided."
        find_text = "\n".join(f"- {f}" for f in findings) if findings else "None provided."

        items_text = "\n\n".join(
            f"ITEM {i+1} [source: {getattr(item, 'source_type', 'unknown')}]\n"
            f"{getattr(item, 'content', '')[:1400]}..."
            for i, item in enumerate(relevant_items[:8])
        )

        existing_text = ""
        if existing_concepts and len(existing_concepts) > 0:
            existing_text = "EXISTING CONCEPTS (refine, extend, or link to these when relevant):\n" + "\n".join(
                f"- {name}: {concept.get('description', '')[:280]}"
                for name, concept in list(existing_concepts.items())[:12]
            )

        return f"""You are a precise computational neuroscientist performing grounded abstraction.

        Task: Extract or refine 5–9 high-quality abstract concepts by analyzing the provided code artifacts against neuroscience theory (Purves et al.).

        You MUST use the following high-level signals to guide your abstraction:
        - Goal: {goal}
        - Hypotheses: 
        {hyp_text}
        - Findings:
        {find_text}

        Rules (strict):
        - Every concept must be a true higher-dimensional abstraction grounded in BOTH the code and the neuroscience principles.
        - Prefer refining or linking to EXISTING concepts when they fit.
        - Create new concepts only when genuinely novel patterns emerge.
        - Avoid fabricating classes, methods, or mechanisms that do not exist in the provided items.
        - Output ONLY a valid JSON array. No extra text.

        {existing_text}

        RELEVANT CODE & BOOK EXCERPTS:
        {items_text}

        Output format (JSON array only):
        [
        {{
            "name": "short, precise name",
            "description": "1-2 sentence grounded mapping between code and neuroscience",
            "type": "mechanism | strategy | principle | analogy",
            "score": 0.75,
            "links_to": ["existing_concept_name_if_relevant"]
        }}
        ]
        """
