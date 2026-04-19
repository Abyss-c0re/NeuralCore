import re
import os
import time
import asyncio
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from neuralcore.utils.prompt_builder import PromptBuilder

from neuralcore.utils.logger import Logger
from neuralcore.utils.search import cosine_sim, keyword_score
from neuralcore.cognition.items import KnowledgeItem

logger = Logger.get_logger()


class KnowledgeConsolidator:
    def __init__(self, agent, max_workers: int = 4):
        self.agent = agent
        self.state = self.agent.state

        self.reranker_model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.concept_graph: Dict[str, Any] = {}

        # Real training data
        self.training_data: List[Tuple[Dict[str, float], int]] = []
        self.min_samples_for_training = 20

        # Simple debounce for heavy extraction
        self._last_extraction_ts = 0.0
        self.extraction_cooldown = 8.0
        self._last_training_ts = 0.0  # seconds between heavy LLM calls

        logger.info("✅ KnowledgeConsolidator initialized")

    # ====================== FEATURE EXTRACTION ======================
    def _extract_features_sync(
        self, query: str, candidates: List[Any], investigation_state: Dict
    ) -> pd.DataFrame:
        rows = []

        fetch_emb = getattr(self.agent.context_manager, "fetch_embedding", None)
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
        if len(candidates) <= k:
            return candidates[:k]

        # Always extract features
        features_df = await asyncio.to_thread(
            self._extract_features_sync,
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
        )

        if self.reranker_model is not None:
            # Use trained model
            scores_raw = await asyncio.to_thread(
                self.reranker_model.predict, features_df
            )
            scores = np.asarray(scores_raw, dtype=np.float64).flatten()
        else:
            # Cold-start hybrid score - completely safe, no .to_numpy / .values on None
            if features_df is None or len(features_df) == 0:
                scores = np.zeros(len(candidates), dtype=np.float64)
            else:
                dense = np.array(features_df.get("dense_cosine", 0.0), dtype=np.float64)
                inv_align = np.array(
                    features_df.get("investigation_align", 0.0), dtype=np.float64
                )
                kw_score = np.array(
                    features_df.get("keyword_score", 0.0), dtype=np.float64
                )
                recency = np.array(
                    features_df.get("recency_score", 0.0), dtype=np.float64
                )

                scores = dense * 0.4 + inv_align * 0.3 + kw_score * 0.2 + recency * 0.1

        scored_items = list(zip(candidates, scores))
        ranked = sorted(scored_items, key=lambda x: float(x[1]), reverse=True)
        reranked_items = [item for item, _ in ranked[:k]]

        # ALWAYS collect training sample
        self._collect_training_sample(query, candidates, reranked_items)

        return reranked_items

    def _collect_training_sample(
        self, query: str, candidates: List[Any], chosen_items: List[Any]
    ):
        """Uses REAL features + gentle semantic boost so dense_cosine finally learns."""
        if len(self.training_data) > 800:
            return

        features_df = self._extract_features_sync(
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
        )

        added = 0
        for i, item in enumerate(candidates):
            if features_df.empty:
                row = {}
            else:
                row = features_df.iloc[i].to_dict()

            # Gentle boost to push higher-dimensional features
            if "dense_cosine" in row:
                row["dense_cosine"] = row.get("dense_cosine", 0.0) * 1.4
            if "investigation_align" in row:
                row["investigation_align"] = row.get("investigation_align", 0.0) * 1.4

            label = 1 if item in chosen_items else 0
            self.training_data.append((row, label))
            added += 1

        pos = sum(1 for _, label in self.training_data if label == 1)
        logger.debug(
            f"[TRAINING DATA] Collected {added} REAL samples | total now: {len(self.training_data)} "
            f"| positive labels: {pos}/{len(self.training_data)}"
        )

        now = time.time()
        if len(self.training_data) >= self.min_samples_for_training:
            last_ts = getattr(self, "_last_training_ts", 0.0)
            if now - last_ts > 60.0:
                self._last_training_ts = now
                logger.info(
                    f"[TRAINING] Starting retrain with {len(self.training_data)} accumulated samples"
                )
                asyncio.create_task(self._schedule_training())

    # ====================== CONCEPT EXTRACTION ======================
    async def extract_and_consolidate(
        self, trace: Optional[List[Dict]] = None, task_goal: str = ""
    ):
        """Main method: distill higher-dimensional abstract concepts + trigger training only on real knowledge growth.
        Keeps NeuralCore abstract — no client logic here."""
        now = time.time()
        if now - self._last_extraction_ts < self.extraction_cooldown:
            logger.debug(
                "[CONSOLIDATE] Cooldown active (%.1fs remaining), skipping heavy extraction",
                self.extraction_cooldown - (now - self._last_extraction_ts),
            )
            return
        self._last_extraction_ts = now

        if trace is None or len(trace) == 0:
            trace = (
                getattr(self.agent.context_manager, "tool_call_history", [])[-20:]
                + getattr(self.agent.context_manager, "action_log", [])[-25:]
            )

        if not trace:
            logger.debug("[CONSOLIDATE] No trace available — skipping")
            return

        goal = getattr(self.agent.state, "task", None) or task_goal or "general task"
        hypotheses = getattr(self.agent.state, "hypotheses", []) or getattr(
            self.agent.context_manager, "investigation_state", {}
        ).get("hypotheses", [])
        findings = getattr(self.agent.state, "findings", []) or getattr(
            self.agent.context_manager, "investigation_state", {}
        ).get("findings", [])

        # Knowledge growth diagnostics
        kb = self.agent.context_manager.knowledge_base
        total_kb_tokens = sum(len(getattr(item, "content", "")) for item in kb.values())
        num_concepts = sum(
            1
            for item in kb.values()
            if getattr(item, "source_type", "") == "extracted_concept"
        )
        num_tool_outcomes = sum(
            1
            for item in kb.values()
            if getattr(item, "source_type", "") == "tool_outcome"
        )

        has_large_item = any(
            getattr(item, "source_type", "") == "tool_outcome"
            and len(getattr(item, "content", "")) > 800
            for item in kb.values()
        )

        logger.debug(
            f"[CONSOLIDATE] KB stats | total_tokens≈{total_kb_tokens:,} | "
            f"concepts={num_concepts} | tool_outcomes={num_tool_outcomes} | large_item={has_large_item}"
        )

        if has_large_item or total_kb_tokens > 15000 or num_concepts >= 5:
            logger.info(
                f"[CONSOLIDATE] Strong knowledge growth detected → "
                f"tokens≈{total_kb_tokens:,} | concepts={num_concepts} | large_item={has_large_item}"
            )
        else:
            logger.debug(
                f"[CONSOLIDATE] Knowledge still small (tokens≈{total_kb_tokens:,}, concepts={num_concepts})"
            )

        logger.info(
            f"[CONSOLIDATE] Starting extraction | goal: {goal[:80]}... | trace len={len(trace)}"
        )

        candidates = list(kb.values())
        logger.debug(
            f"[CONSOLIDATE] Reranking {len(candidates)} candidates for concept distillation"
        )
        relevant = await self.rerank(goal, candidates, k=40)

        extracted = await self._distill_concepts(relevant, goal, hypotheses, findings)

        added = 0
        for concept in extracted:
            item = KnowledgeItem(
                key=f"concept_{int(time.time() * 1000)}_{hash(concept.get('name', '')) % 10000}",
                source_type="extracted_concept",
                content=concept.get("description", ""),
                metadata={
                    "type": concept.get("type", "strategy"),
                    "confidence": concept.get("score", 0.82),
                    "goal": goal,
                    "from_large_item": has_large_item,
                },
            )

            if hasattr(self.agent.context_manager, "fetch_embedding"):
                item.embedding = await self.agent.context_manager.fetch_embedding(
                    concept.get("description", ""), prefix="passage"
                )

            self.agent.context_manager.knowledge_base[item.key] = item
            added += 1

        if added > 0:
            if hasattr(self.agent.context_manager, "_sparse_index_dirty"):
                self.agent.context_manager._sparse_index_dirty = True
            logger.info(
                f"✅ Stored {added} new abstract concepts (large item: {has_large_item})"
            )
        else:
            logger.debug("[CONSOLIDATE] No new concepts distilled this round")

        self._update_concept_graph(extracted)

        # Smart training trigger — only on meaningful growth
        if (
            added >= 2
            or has_large_item
            or total_kb_tokens > 25000
            or num_concepts >= 12
        ):
            logger.info(
                f"[TRAINING TRIGGER] Meaningful growth detected → scheduling retrain "
                f"(added={added}, tokens≈{total_kb_tokens:,}, concepts={num_concepts}, large={has_large_item})"
            )
            asyncio.create_task(self._schedule_training())
        else:
            logger.debug(
                f"[TRAINING TRIGGER] Growth too small for full retrain (added={added}, tokens≈{total_kb_tokens:,}, concepts={num_concepts})"
            )

    async def _distill_concepts(
        self, relevant_items: List[Any], goal: str, hypotheses: List, findings: List
    ) -> List[Dict]:
        if not relevant_items:
            return []

        prompt = PromptBuilder.abstract_concept_extraction(
            goal=goal,
            hypotheses=hypotheses,
            findings=findings,
            relevant_items=relevant_items,
        )

        try:
            response = await self.agent.client.ask(
                prompt, temperature=0.28, max_tokens=1400
            )
            import re

            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE
            )
            concepts = json.loads(cleaned)
            if isinstance(concepts, list):
                logger.info(
                    f"[DISTILL] Successfully extracted {len(concepts)} abstract concepts"
                )
                return concepts
        except Exception as e:
            logger.error(f"[DISTILL] LLM extraction failed: {e}")

        return []

    def _update_concept_graph(self, concepts: List[Dict]):
        for c in concepts:
            name = c.get("name")
            if name:
                self.concept_graph[name] = c

    # ====================== TRAINING ======================
    async def _schedule_training(self):
        """Accumulates data across runs (no full clear) — this is what allows the model to grow richer over time."""
        current_samples = len(self.training_data)
        if current_samples < self.min_samples_for_training:
            logger.debug(
                f"[TRAINING] Only {current_samples}/{self.min_samples_for_training} samples — waiting"
            )
            return

        logger.info(
            f"[TRAINING] Starting retrain with {current_samples} accumulated samples"
        )

        try:
            X_list = [sample[0] for sample in self.training_data]
            y = np.array([sample[1] for sample in self.training_data])
            X = pd.DataFrame(X_list)

            groups = [len(X)] if len(X) > 0 else [1]

            await self.train_reranker(X, y, groups)

            # Keep recent history for diversity
            if len(self.training_data) > 500:
                self.training_data = self.training_data[-500:]
            logger.info(
                f"[TRAINING] Kept last {len(self.training_data)} samples for future runs"
            )

        except Exception as e:
            logger.error(f"[TRAINING] Failed during scheduling: {e}", exc_info=True)

    def _get_model_path(self) -> str:
        app_config = getattr(self.agent.loader, "config", {}).get("app", {})
        models_dir = app_config.get("models_dir", "~/.neuralcore/models")
        path = Path(os.path.expanduser(models_dir)).resolve()
        path.mkdir(parents=True, exist_ok=True)
        agent_id = getattr(self.agent, "agent_id", "default")
        return str(path / f"{agent_id}_knowledge_consolidator_ltr.txt")

    async def train_reranker(self, X: pd.DataFrame, y: np.ndarray, groups: List[int]):
        """Train LambdaMART with stronger condensation + semantic bias."""
        if len(X) < 10:
            logger.warning("Too few samples to train reranker")
            return

        model_path = self._get_model_path()
        logger.info(f"Training LambdaMART on {len(X)} samples → {model_path}")

        try:
            train_set = lgb.Dataset(X, label=y, group=groups)

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "learning_rate": 0.03,          # slower, more stable
                "num_leaves": 20,               # tighter cap
                "min_data_in_leaf": 12,         # stronger generalization
                "boosting_type": "gbdt",
                "feature_fraction": 0.80,
                "bagging_fraction": 0.80,
                "bagging_freq": 5,
                "verbose": -1,
                "lambda_l1": 0.5,               # new: light L1 regularization
                "lambda_l2": 0.8,               # new: light L2
            }

            self.reranker_model = lgb.train(
                params,
                train_set,
                num_boost_round=500,
                valid_sets=[train_set],
                callbacks=[
                    lgb.early_stopping(80, verbose=False),   # stronger stopping
                    lgb.log_evaluation(0),
                ],
            )

            self.feature_names = list(X.columns)

            num_trees = getattr(
                self.reranker_model,
                "num_trees",
                lambda: len(getattr(self.reranker_model, "trees", [])),
            )()

            importances = dict(
                zip(self.feature_names, self.reranker_model.feature_importance())
            )

            # Extra condensation: if too many trees, we can log and optionally re-train stricter next time
            if num_trees > 180:
                logger.warning(
                    f"Model reached {num_trees} trees — condensation working but monitor growth. "
                    f"Consider increasing min_data_in_leaf or lambda_l1/l2 if dense_cosine stays low."
                )

            self.reranker_model.save_model(model_path)

            logger.info(
                f"✅ LambdaMART model trained and saved | trees={num_trees} | "
                f"feature_importances: {importances}"
            )

        except Exception as e:
            logger.error(f"Failed to train reranker: {e}", exc_info=True)

    async def load_reranker(self):
        model_path = self._get_model_path()
        try:
            import os

            if not os.path.exists(model_path):
                logger.info(
                    f"No trained model at {model_path}. Will bootstrap from hybrid scores."
                )
                self.reranker_model = None
                return

            self.reranker_model = await asyncio.to_thread(
                lgb.Booster, model_file=model_path
            )
            self.feature_names = self.reranker_model.feature_name()
            logger.info(
                f"✅ Loaded LambdaMART reranker ({len(self.feature_names)} features)"
            )
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            self.reranker_model = None
