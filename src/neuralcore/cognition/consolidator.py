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

        # ====================== NOVELTY DETECTION ======================
        # Configurable similarity threshold to skip already-known concepts
        # Can be set in config under: app.cognition.novelty_threshold
        app_config = getattr(getattr(self.agent, "loader", None), "config", {}).get(
            "app", {}
        )
        cognition_config = app_config.get("cognition", {})
        self.novelty_threshold: float = cognition_config.get("novelty_threshold", 0.85)

        # Real training data
        self.training_data: List[Tuple[Dict[str, float], int]] = []
        self.min_samples_for_training = 10

        # Simple debounce for heavy extraction
        self._last_extraction_ts = 0.0
        self.extraction_cooldown = 8.0
        self._last_training_ts = 0.0

        logger.info(
            f"✅ KnowledgeConsolidator initialized | novelty_threshold={self.novelty_threshold}"
        )

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

        features_df = await asyncio.to_thread(
            self._extract_features_sync,
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
        )

        if self.reranker_model is not None:
            scores_raw = await asyncio.to_thread(
                self.reranker_model.predict, features_df
            )
            scores = np.asarray(scores_raw, dtype=np.float64).flatten()
        else:
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

        self._collect_training_sample(query, candidates, reranked_items)

        return reranked_items

    def _collect_training_sample(
        self, query: str, candidates: List[Any], chosen_items: List[Any]
    ):
        if len(self.training_data) > 800:
            return

        features_df = self._extract_features_sync(
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
        )

        added = 0
        for i, item in enumerate(candidates):
            row = features_df.iloc[i].to_dict() if not features_df.empty else {}

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
        """Main method: distill higher-dimensional abstract concepts.
        Keeps NeuralCore fully abstract and reusable for any domain."""
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
                getattr(self.agent.context_manager, "tool_call_history", [])[-30:]
                + getattr(self.agent.context_manager, "action_log", [])[-30:]
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

        kb = self.agent.context_manager.knowledge_base
        total_kb_tokens = sum(len(getattr(item, "content", "")) for item in kb.values())
        num_concepts = sum(
            1
            for item in kb.values()
            if getattr(item, "source_type", "") == "extracted_concept"
        )
        has_large_item = any(
            getattr(item, "source_type", "") == "tool_outcome"
            and len(getattr(item, "content", "")) > 800
            for item in kb.values()
        )

        logger.debug(
            f"[CONSOLIDATE] KB stats | tokens≈{total_kb_tokens:,} | concepts={num_concepts} | large_item={has_large_item}"
        )
        logger.info(
            f"[CONSOLIDATE] Starting extraction | goal: {goal[:80]}... | trace len={len(trace)}"
        )

        candidates = list(kb.values())
        relevant = await self.rerank(goal, candidates, k=60)

        # === NOVELTY FILTER BEFORE DISTILLATION ===
        novel_relevant = self._filter_novel_items(relevant)

        if not novel_relevant:
            logger.debug(
                "[CONSOLIDATE] All candidates already known — skipping distillation"
            )
            return

        extracted = await self._distill_concepts(
            novel_relevant,
            goal,
            hypotheses,
            findings,
            self.concept_graph,
        )

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
            logger.info(f"✅ Stored {added} new abstract concepts")
        else:
            logger.debug("[CONSOLIDATE] No new concepts distilled this round")

        self._update_concept_graph(extracted)

        if added >= 1 or has_large_item or total_kb_tokens > 20000 or num_concepts >= 8:
            logger.info(
                f"[TRAINING TRIGGER] Meaningful growth detected → scheduling retrain "
                f"(added={added}, tokens≈{total_kb_tokens:,}, concepts={num_concepts})"
            )
            asyncio.create_task(self._schedule_training())
        else:
            logger.debug(
                f"[TRAINING TRIGGER] Growth too small for full retrain (added={added})"
            )

    # ====================== NEW: NOVELTY FILTER ======================
    def _filter_novel_items(self, candidates: List[Any]) -> List[Any]:
        """Return only items that are sufficiently novel compared to existing concepts."""
        if not self.concept_graph or not candidates:
            return candidates

        novel = []
        threshold = self.novelty_threshold

        for item in candidates:
            item_emb = getattr(item, "embedding", None)
            if item_emb is None or len(item_emb) == 0:
                novel.append(item)
                continue

            max_sim = 0.0
            for concept in self.concept_graph.values():
                concept_emb = concept.get("embedding")
                if concept_emb is not None and len(concept_emb) > 0:
                    sim = cosine_sim(item_emb, concept_emb)
                    if sim > max_sim:
                        max_sim = sim

            if max_sim < threshold:
                novel.append(item)

        logger.debug(
            f"[NOVELTY] Filtered {len(candidates) - len(novel)} known items "
            f"(threshold={threshold}) → {len(novel)} novel candidates"
        )
        return novel

    async def _distill_concepts(
        self,
        relevant_items: List[Any],
        goal: str,
        hypotheses: List,
        findings: List,
        existing_concepts: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        if not relevant_items:
            logger.debug("[DISTILL] No relevant items — skipping")
            return []

        prompt = PromptBuilder.abstract_concept_extraction(
            goal=goal,
            hypotheses=hypotheses,
            findings=findings,
            relevant_items=relevant_items,
            existing_concepts=existing_concepts,
        )

        try:
            response = await self.agent.client.ask(
                prompt, temperature=0.28, max_tokens=1400
            )
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
                # Safely handle None values for description and content
                description = str(c.get("description") or "").strip()
                # Find the corresponding KnowledgeItem that was just created
                matching_item = next(
                    (
                        item
                        for item in self.agent.context_manager.knowledge_base.values()
                        if getattr(item, "key", "").startswith("concept_")
                        and description in str(getattr(item, "content", "")).strip()
                    ),
                    None,
                )
                if (
                    matching_item
                    and getattr(matching_item, "embedding", None) is not None
                ):
                    c["embedding"] = matching_item.embedding
                self.concept_graph[name] = c

    # ====================== TRAINING (unchanged) ======================
    async def _schedule_training(self):
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

            if len(self.training_data) > 800:
                self.training_data = self.training_data[-600:]

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
        if len(X) < 10:
            logger.warning("Too few samples to train reranker")
            return

        model_path = self._get_model_path()
        logger.info(
            f"Training LambdaMART on {len(X)} samples → {model_path} (incremental)"
        )

        try:
            init_model = None
            if os.path.exists(model_path):
                try:
                    init_model = lgb.Booster(model_file=model_path)
                    logger.debug("Loaded existing model for incremental training")
                except Exception as e:
                    logger.warning(
                        f"Could not load existing model: {e}. Starting fresh."
                    )

            train_set = lgb.Dataset(X, label=y, group=groups)

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "learning_rate": 0.06,
                "num_leaves": 32,
                "min_data_in_leaf": 6,
                "boosting_type": "gbdt",
                "feature_fraction": 0.90,
                "bagging_fraction": 0.85,
                "bagging_freq": 3,
                "verbose": -1,
                "lambda_l1": 0.1,
                "lambda_l2": 0.2,
            }

            self.reranker_model = lgb.train(
                params,
                train_set,
                num_boost_round=300,
                valid_sets=[train_set],
                init_model=init_model,
                callbacks=[
                    lgb.early_stopping(40, verbose=False),
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

            if num_trees > 250:
                logger.info(
                    f"Model has {num_trees} trees — still growing naturally. "
                    f"Monitor dense_cosine / investigation_align rising."
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

    def reset_model(self, keep_graph: bool = True):
        logger.info("🔄 Resetting LambdaMART model due to prompt/extraction changes")

        self.training_data.clear()
        self.reranker_model = None
        self.feature_names = []

        if not keep_graph:
            self.concept_graph.clear()
            logger.info("Concept graph also cleared")
        else:
            logger.info(
                f"Kept existing concept_graph with {len(self.concept_graph)} concepts"
            )

        model_path = self._get_model_path()
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                logger.info(f"Deleted old model file: {model_path}")
            except Exception as e:
                logger.warning(f"Could not delete model file: {e}")

        logger.info("✅ Model reset complete. Next training will start from scratch.")
