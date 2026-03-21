from neuralcore.actions.actions import Action, ActionSet, sequence, ActionFromSequence


async def check_goal_action(agent, **kwargs):
    """
    Check if the agent's goal has been achieved.
    If yes, mark sequence as completed and exit early.
    """
    # Evaluate agent state: could be custom method like agent.is_goal_achieved()
    goal_achieved = getattr(agent, "is_goal_achieved", lambda: False)()

    if goal_achieved:
        return {
            "stop": True,
            "reason": "goal_achieved",
            "marker": "[FINAL_ANSWER_COMPLETE]",
        }

    return {"next_step": "analyze_tools"}


async def analyze_tools_action(input: dict, **kwargs):
    full_reply = input.get("full_reply", "")
    tool_calls = input.get("tool_calls", [])

    if "[FINAL_ANSWER_COMPLETE]" in full_reply:
        return {"stop": True, "reason": "final_answer_detected"}

    if any(tc.get("status") == "pending" for tc in tool_calls):
        return {"next_step": "execute_tools"}

    return {"next_step": "reflect_if_stuck"}


async def reflect_if_needed_action(input: dict, **kwargs):
    next_step = input.get("next_step")
    if next_step == "reflect_if_stuck":
        return {"next_step": "reflect_if_stuck"}
    return input


async def choose_next_step_action(input: dict, **kwargs):
    return {
        "next_step": input.get("next_step", "llm_stream"),
        "stop": input.get("stop", False),
        "tool_to_run": input.get("tool_to_run"),
    }


def add_decisions(agent):
    """
    Registers autonomous decision-making actions for an agent.
    Pass the agent object so steps can access its state.
    """

    # Wrap agent-aware steps
    check_goal = Action(
        name="check_goal",
        description="Check if agent's goal is achieved; exit early if so",
        parameters={"agent": {"type": "object", "description": "Agent instance"}},
        executor=lambda **kwargs: check_goal_action(agent, **kwargs),
    )

    analyze_action = Action(
        name="analyze_tools",
        description="Analyze iteration output and determine next step",
        parameters={
            "input": {
                "type": "object",
                "description": "Agent iteration state (full_reply, tool_calls)",
            }
        },
        executor=analyze_tools_action,
    )

    reflect_action = Action(
        name="reflect_if_needed",
        description="Decide if reflection is needed",
        parameters={"input": {"type": "object", "description": "Analysis result"}},
        executor=reflect_if_needed_action,
    )

    choose_action = Action(
        name="choose_next_step",
        description="Pick next workflow step or stop",
        parameters={"input": {"type": "object", "description": "Reflection result"}},
        executor=choose_next_step_action,
    )

    # Create a sequence including goal check and optional reset
    decision_seq = sequence(
        name="decision_maker",
        description="Autonomous workflow sequence with agent goal evaluation",
        steps=[check_goal, analyze_action, reflect_action, choose_action],
    )

    # Wrap sequence as Action for registry
    decision_action = ActionFromSequence.create(decision_seq, name="decision_maker")

    # Add to ActionSet
    decision_set = ActionSet(
        name="DecisionTools",
        description="Autonomous decision-making actions for the agent",
        actions=[decision_action],
    )

    agent.registry.register_set("DecisionTools", decision_set)
