from datetime import datetime

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
