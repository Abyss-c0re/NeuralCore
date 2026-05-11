from enum import Enum

from neuralcore.agents.state import AgentState
from neuralcore.workflows.registry import workflow
from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder
from neuralcore.tasks.helpers import classify_intent

logger = Logger.get_logger()


# ===================================================================
# CONDITIONS (only those actively used by TaskManager + loops)
# ===================================================================
@workflow.condition("goal_achieved")
def goal_achieved(state: AgentState, args=None) -> bool:
    """Core condition used by goal_driven_loop / pure_goal_driven."""
    if getattr(state, "mode", None) == "casual":
        return False

    full_reply = getattr(state, "full_reply", "").strip()
    is_complete = getattr(state, "is_complete", False)
    goal_achieved_flag = getattr(state, "goal_achieved", False)

    has_real_content = len(full_reply) > 50 and not any(
        w in full_reply.lower()
        for w in ["error", "failed", "try again", "still working", "reading file again"]
    )

    explicit_done = any(
        phrase in full_reply.lower()
        for phrase in [
            "task complete",
            "goal achieved",
            "all done",
            "finished successfully",
        ]
    )

    all_subtasks_done = (
        len(getattr(state, "planned_tasks", [])) == 0
        or getattr(state, "current_task_index", 0)
        >= len(getattr(state, "planned_tasks", [])) - 1
    )

    marker = PromptBuilder.FINAL_ANSWER_MARKER
    has_marker_in_reply = marker in full_reply
    if has_marker_in_reply:
        state.full_reply = full_reply.replace(marker, "").strip()

    last_success = getattr(state, "last_tool_success", None)
    tool_reported_success = bool(last_success and last_success.get("success"))

    return (
        (is_complete or goal_achieved_flag)
        and has_real_content
        and all_subtasks_done
        and (explicit_done or has_marker_in_reply or tool_reported_success)
    )


@workflow.condition("too_many_empty_loops")
def too_many_empty_loops(state: AgentState, args=None):
    """Used by TaskManager for empty loop safety."""
    max_allowed = args.get("max", 5) if isinstance(args, dict) else 5
    return getattr(state, "empty_loops", 0) >= max_allowed


@workflow.condition("max_action_restarts_reached")
def max_action_restarts_reached(state: AgentState, args=None):
    """Used by TaskManager for action restart safety."""
    max_allowed = args.get("max", 8) if isinstance(args, dict) else 8
    return getattr(state, "action_restarts", 0) >= max_allowed


# ===================================================================
# LOOPS (core reusable orchestration units)
# ===================================================================
@workflow.loop("goal_driven_loop", max_iterations=None, break_condition="goal_achieved")
async def goal_driven_loop(agent, state: AgentState):
    """Inner goal-driven loop – fully delegated to TaskManager."""
    yield ("phase_changed", {"phase": "thinking"})
    async for ev, pl in agent.task_manager.run_goal_driven_loop(
        state, "goal_driven_loop"
    ):
        yield ev, pl


@workflow.loop("chat_tool_loop", max_iterations=None)
async def chat_tool_loop(agent, state: AgentState):
    """
    Persistent outer loop for interactive/chat agents only.
    - CASUAL → lightweight reply
    - TASK → one-time planning + goal_driven_loop
    """
    logger.info("[CHAT TOOL LOOP] Outer loop started — persistent mode active")

    target_loop = "chat_tool_loop"
    content = await agent.wait_for_incoming_message(
        role="user", return_content_only=True
    )

    intent = await classify_intent(agent, content)
    logger.debug(f"Intent {intent}")

    if content:
        if intent == "CASUAL":
            logger.info("[CASUAL MODE] Pure basic chat — no inner loop")
            yield ("phase_changed", {"phase": "casual_chat"})

            casual_messages = await agent.context_manager.provide_context(
                query=content,
                max_input_tokens=agent.max_tokens * 0.65,
                reserved_for_output=agent.client.max_tokens * 0.35,
                include_logs=False,
                chat=True,
            )

            final_reply = await agent.client.chat(
                casual_messages,
                temperature=0.85,
                top_p=0.95,
                max_tokens=agent.client.max_tokens * 0.4,
            )

            await agent.add_message("assistant", final_reply)
            yield ("llm_response", {"full_reply": final_reply, "is_complete": True})

            agent.message_queue.task_done()
            state.reset_for_new_task()

            state.request_loop_restart(
                reason="Casual chat completed, waiting for next message",
                target_loop=target_loop,
            )
        else:
            # TASK-DRIVEN MODE
            logger.info("[TASK-DRIVEN MODE] Delegating to goal_driven_loop")
            yield ("phase_changed", {"phase": "goal_driven"})
            state.reset_for_new_task(new_task=content)

            if not state.planned_tasks:
                async for event, payload in agent.task_manager.plan():
                    yield event, payload

            async for event, payload in agent.execute_loop(
                "goal_driven_loop", initial_state=state
            ):
                yield event, payload

                if event in ("error", "cancelled", "loop_stopped"):
                    state.request_loop_stop(
                        reason=f"Inner loop signaled {event}", target_loop=target_loop
                    )
                    return

            logger.info("[TASK-DRIVEN MODE] goal_driven_loop completed")

            if state.goal_reached:
                yield ("phase_changed", {"phase": "generating_final_answer"})
                results = await agent.context_manager.provide_context(
                    query=content,
                    lightweight_agentic=True,
                    max_input_tokens=agent.client.max_tokens * 0.65,
                    reserved_for_output=agent.client.max_tokens * 0.4,
                    return_as_string=True,
                )
                final_reply = await agent.client.chat(
                    results,
                    temperature=0.0,
                    top_p=0.1,
                    max_tokens=agent.client.max_tokens,
                )
                yield (
                    "llm_response",
                    {"full_reply": final_reply, "tool_calls": [], "is_complete": True},
                )
                state.reset_for_new_task()
            else:
                state.request_loop_restart(
                    reason="Fallback restart", target_loop=target_loop
                )
                yield ("phase_changed", {"phase": "restarting_loop"})

            if not state.goal_reached:
                state.status = "idle"
                state.is_complete = True

            state.request_loop_restart(
                reason="Inner goal_driven_loop finished, returning to chat",
                target_loop=target_loop,
            )


@workflow.loop("pure_goal_driven", max_iterations=None, break_condition="goal_achieved")
async def pure_goal_driven(agent, state: AgentState):
    """
    Headless / non-chat goal-driven mode.
    No message waiting, no casual branch – pure TaskManager execution.
    """
    logger.info("[PURE GOAL-DRIVEN] Starting headless/task-only execution")
    yield ("phase_changed", {"phase": "goal_driven"})

    if not state.planned_tasks:
        async for event, payload in agent.task_manager.plan():
            yield event, payload

    async for event, payload in agent.execute_loop(
        "goal_driven_loop", initial_state=state
    ):
        yield event, payload

        if event in ("error", "cancelled", "loop_stopped"):
            return

    logger.info("[PURE GOAL-DRIVEN] Completed successfully")


# ===================================================================
# AGENT FLOWS (minimal, reusable)
# ===================================================================
class AgentFlow:
    class Phase(str, Enum):
        IDLE = "idle"
        CHAT = "chat"
        GOAL_DRIVEN = "goal_driven"

    def __init__(self, agent):
        self.agent = agent
        self.engine = agent.workflow
        workflow.bind_to_engine(self.engine, instance=self)

    @workflow.step("deploy_agent", name="deploy_agent_loop")
    async def _wf_deploy_agent_loop(self, iteration: int, state: AgentState):
        """Persistent chat mode (interactive agents)."""
        if iteration == 0:
            state.phase = self.Phase.CHAT
            yield ("phase_changed", {"phase": "chat"})
            logger.info(f"Agent '{self.agent.name}' → Chat mode started")

        async for ev, pl in self.agent.execute_loop(
            "chat_tool_loop", initial_state=state
        ):
            yield ev, pl

