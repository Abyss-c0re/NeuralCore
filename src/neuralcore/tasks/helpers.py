from neuralcore.utils.logger import Logger
from neuralcore.utils.prompt_builder import PromptBuilder

logger = Logger.get_logger()


async def classify_intent(agent, query: str) -> str:
    if not query or not query.strip():
        return "CASUAL"
    try:
        result = await agent.client.chat(
            PromptBuilder.classify_intent(query), temperature=0.0, max_tokens=50
        )
        cleaned = result.strip().upper()
        return "CASUAL" if "CASUAL" in cleaned else "TASK"
    except Exception as e:
        logger.warning(f"classify_intent failed, falling back: {e}")
        return "CASUAL" if len(query.split()) < 25 else "TASK"

