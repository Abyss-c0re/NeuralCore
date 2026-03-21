# 1. Register a custom workflow (anywhere — plugin, config, runtime)
engine.register_workflow(
    name="my_advanced_react",
    description="ReAct + extra reflection + safety checks + my custom tool step",
    steps=[
        "plan_tasks",
        "llm_stream",
        "execute_if_tools",
        "my_custom_step",           # ← external
        "verify_goal_completion",
        "reflect_if_stuck",
        {"name": "safety_fallback", "overrides": {"temperature": 0.3}},
    ]
)

# 2. (Optional) Add external step
engine.register_step("my_custom_step", my_external_handler)

# 3. Tell agent to use it
agent.config["workflow"] = {"name": "my_advanced_react"}



or

import sequence, ActionFromSequence

# Example: create a reusable sequence once
research_sequence = sequence(
    name="deep_research_chain",
    description="Search web → summarize key facts → fact-check → compile report",
    steps=[
        web_search_action,
        summarize_action,
        fact_check_action,
        report_writer_action,
    ],
    propagate=True,
    output_from=-1,                    # last step's output
    confirm_predicate=lambda r: "price" in str(r).lower() or len(str(r)) > 4000
)

# Wrap it so it looks like any other tool
research_tool = ActionFromSequence.create(
    research_sequence,
    name="deep_research",
    description=research_sequence.description,
    tags=["research", "multi-step", "report"],
)


engine = WorkflowEngine(agent)

# Register the wrapped sequence as if it were a normal step
engine.register_step("deep_research", research_tool)   # ← note: it's now a callable Action

# Then include it in a named workflow
engine.register_workflow(
    name="research_heavy",
    description="Agent that prefers multi-step research tools over single calls",
    steps=[
        "plan_tasks",
        "llm_stream",
        "deep_research",               # ← your sequence appears here as one atomic step
        "execute_if_tools",
        "verify_goal_completion",
        "reflect_if_stuck",
    ]
)

or 

engine.register_step("web_search_phase", web_search_action)
engine.register_step("summarize_phase", summarize_action)
engine.register_step("fact_check_phase", fact_check_action)
engine.register_step("report_phase", report_writer_action)

# Then define a workflow that uses them in order
engine.register_workflow(
    name="explicit_research",
    description="Fine-grained research with possible early exit or reflection",
    steps=[
        "plan_tasks",
        "llm_stream",
        "web_search_phase",
        "summarize_phase",
        {"name": "fact_check_phase", "overrides": {"temperature": 0.2}},
        "report_phase",
        "verify_goal_completion",
        "reflect_if_stuck_if_no_progress",
    ]
)