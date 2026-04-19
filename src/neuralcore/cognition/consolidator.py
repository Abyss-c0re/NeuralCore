import asyncio
import time
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from neuralcore.utils.logger import Logger
from neuralcore.utils.search import cosine_sim, keyword_score
from neuralcore.cognition.items import KnowledgeItem

logger = Logger.get_logger()


class KnowledgeConsolidator:
    def __init__(self, agent, max_workers: int = 4):
        """
        agent: Full Agent instance (gives access to context_manager + state)
        """
        self.agent = agent
        self.context = agent.context_manager
        self.state = agent.state

        self.reranker_model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.concept_graph: Dict[str, Any] = {}

    # ====================== FEATURE EXTRACTION ======================
    def _extract_features_sync(
        self, query: str, candidates: List[Any], investigation_state: Dict
    ) -> pd.DataFrame:
        rows = []

        fetch_emb = getattr(self.context, "fetch_embedding", None)
        query_emb = fetch_emb(query, prefix="query") if fetch_emb else None

        inv_summary = (
            str(investigation_state.get("goal", ""))
            + " "
            + " ".join(investigation_state.get("hypotheses", []))
        )
        inv_emb = (
            fetch_emb(inv_summary, prefix="query")
            if fetch_emb and inv_summary.strip()
            else None
        )

        for item in candidates:
            recency_seconds = time.time() - getattr(item, "timestamp", time.time())
            recency_hours = recency_seconds / 3600.0

            emb = getattr(item, "embedding", None)

            row = {
                "dense_cosine": self._safe_cosine(query_emb, emb),
                "investigation_align": self._safe_cosine(inv_emb, emb),
                "keyword_score": keyword_score(
                    query_words=re.findall(r"\b\w+\b", query.lower()),
                    text=getattr(item, "content", ""),
                ),
                "recency_score": np.exp(-0.75 * recency_hours),
                "recency_minutes": recency_seconds / 60.0,
                "source_type_score": self._encode_source_type(
                    getattr(item, "source_type", "")
                ),
                "content_length": len(getattr(item, "content", "")),
                "is_tool_outcome": 1.0
                if getattr(item, "source_type", "") == "tool_outcome"
                else 0.0,
                "is_concept": 1.0
                if getattr(item, "source_type", "") == "extracted_concept"
                else 0.0,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0.0)
        return df

    def _safe_cosine(
        self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]
    ) -> float:
        if emb1 is None or emb2 is None:
            return 0.0
        try:
            return cosine_sim(emb1, emb2)
        except Exception:
            return 0.0

    def _encode_source_type(self, st: str) -> float:
        mapping = {
            "extracted_concept": 3.0,
            "tool_outcome": 2.0,
            "agent_trace": 2.5,
            "raw_knowledge": 1.0,
        }
        return mapping.get(str(st).lower(), 1.0)

    # ====================== RERANKING ======================
    async def rerank(self, query: str, candidates: List[Any], k: int = 20) -> List[Any]:
        if not self.reranker_model or len(candidates) <= k:
            return candidates[:k]

        features_df = await asyncio.to_thread(
            self._extract_features_sync,
            query,
            candidates,
            self.state.__dict__,
        )

        scores_raw = await asyncio.to_thread(self.reranker_model.predict, features_df)
        scores = np.asarray(scores_raw, dtype=np.float64).flatten()

        scored_items = list(zip(candidates, scores))
        ranked = sorted(scored_items, key=lambda x: float(x[1]), reverse=True)

        return [item for item, _ in ranked[:k]]

    # ====================== CONCEPT EXTRACTION ======================
    async def extract_and_consolidate(self, trace: List[Dict], task_goal: str = ""):
        if not trace:
            return

        goal = getattr(self.state, "task", None) or task_goal or "general task"
        hypotheses = getattr(
            self.state, "hypotheses", []
        ) or self.context.investigation_state.get("hypotheses", [])
        findings = getattr(
            self.state, "findings", []
        ) or self.context.investigation_state.get("findings", [])

        candidates = list(self.context.knowledge_base.values())

        relevant = await self.rerank(goal, candidates, k=30)

        extracted = await self._distill_concepts(relevant, goal, hypotheses, findings)

        added = 0
        for concept in extracted:
            item = KnowledgeItem(
                key=f"concept_{concept.get('name', int(time.time()))}",
                source_type="extracted_concept",
                content=concept.get("description", ""),
                metadata={
                    "type": concept.get("type", "strategy"),
                    "confidence": concept.get("score", 0.75),
                    "goal": goal,
                    "from_state": True,
                },
            )

            if hasattr(self.context, "fetch_embedding"):
                item.embedding = await self.context.fetch_embedding(
                    concept.get("description", ""), prefix="passage"
                )

            self.context.knowledge_base[item.key] = item
            added += 1

        if added > 0:
            if hasattr(self.context, "_sparse_index_dirty"):
                self.context._sparse_index_dirty = True
            logger.info(
                f"KnowledgeConsolidator: Extracted and stored {added} abstract concepts"
            )

        self._update_concept_graph(extracted)

    async def _distill_concepts(
        self, relevant_items, goal: str, hypotheses: List, findings: List
    ) -> List[Dict]:
        """Placeholder — replace with real LLM call using PromptBuilder + agent.client"""
        return []

    def _update_concept_graph(self, concepts: List[Dict]):
        for c in concepts:
            name = c.get("name")
            if name:
                self.concept_graph[name] = c

    # ====================== TRAINING & LOADING ======================
    async def train_reranker(self, X: pd.DataFrame, y: np.ndarray, groups: List[int]):
        # TODO: implement when training data is collected
        pass

    async def load_reranker(
        self, path: str = "neuralcore_knowledge_consolidator_ltr.txt"
    ):
        # TODO: implement model persistence
        pass
