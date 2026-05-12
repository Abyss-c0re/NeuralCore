"""
Mock tools that produce structured context data.
These tools are designed to be loaded by the agent and their outputs
will populate short_term_mem in the ContextManager.
"""

from neuralcore.actions.registry import tool


@tool(
    "ContextTestTools",
    tags=["context", "weather", "data"],
    name="get_weather_data",
    description="Retrieve current weather data for a given city.",
)
async def get_weather_data(city: str) -> str:
    return (
        f"Weather Report for {city}:\n"
        f"Temperature: 22°C (72°F)\n"
        f"Humidity: 65%\n"
        f"Wind: 12 km/h NW\n"
        f"Conditions: Partly cloudy\n"
        f"UV Index: 6 (High)\n"
        f"Air Quality Index: 42 (Good)\n"
        f"Forecast: Clearing skies expected by evening."
    )


@tool(
    "ContextTestTools",
    tags=["context", "system", "info"],
    name="get_system_metrics",
    description="Retrieve system performance metrics and resource utilization.",
)
async def get_system_metrics() -> str:
    return (
        "System Metrics Report:\n"
        "CPU Usage: 34.2% (8 cores)\n"
        "Memory: 12.4 GB / 32 GB (38.8%)\n"
        "Disk I/O: Read 245 MB/s, Write 180 MB/s\n"
        "Network: Inbound 520 Mbps, Outbound 340 Mbps\n"
        "Active Processes: 287\n"
        "Uptime: 14 days, 6 hours, 32 minutes\n"
        "Load Average: 2.14, 1.87, 1.62"
    )


@tool(
    "ContextTestTools",
    tags=["context", "project", "status"],
    name="get_project_status",
    description="Get the current project status including milestones and risks.",
)
async def get_project_status(project_name: str) -> str:
    return (
        f"Project Status: {project_name}\n"
        f"Phase: Development Sprint 4\n"
        f"Completion: 67%\n"
        f"Milestone: API v2 integration due 2025-02-15\n"
        f"Open Issues: 23 (5 critical, 8 high, 10 medium)\n"
        f"Team Velocity: 42 story points/sprint\n"
        f"Risk: Database migration deadline at risk due to schema changes\n"
        f"Budget Utilization: 58% of allocated funds\n"
        f"Next Review: Weekly standup on Monday 09:00 UTC"
    )
