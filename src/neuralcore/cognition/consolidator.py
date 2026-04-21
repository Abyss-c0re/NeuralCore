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
        app_config = getattr(getattr(self.agent, "loader", None), "config", {}).get(
            "app", {}
        )
        cognition_config = app_config.get("cognition", {})
        self.novelty_threshold: float = cognition_config.get("novelty_threshold", 0.85)

        # Training data
        self.training_data: List[Tuple[Dict[str, float], int]] = []
        self.min_samples_for_training = 10

        # Debounce
        self._last_extraction_ts = 0.0
        self.extraction_cooldown = 8.0
        self._last_training_ts = 0.0

        self.active_features: List[str] = [
            "keyword_score",
            "content_length",
            "source_type_score",
            "is_tool_outcome",
            "recency_score",
            "dense_cosine",
            "cosine_x_keyword",
            "semantic_rescue",
            "kw_x_length",
            "tool_x_source",
            "category_score",  # NEW: helps the model prefer certain categories
        ]

        logger.info(
            f"✅ KnowledgeConsolidator initialized | novelty_threshold={self.novelty_threshold}"
        )

    # ====================== NEW: Get candidates from both sources ======================
    def _get_all_candidates(self) -> List[Any]:
        """Combine short-term memory + persistent KnowledgeBase items."""
        candidates: List[Any] = []

        # 1. Short-term memory (existing behavior)
        short_term = getattr(self.agent.context_manager, "knowledge_base", {})
        if isinstance(short_term, dict):
            candidates.extend(list(short_term.values()))

        # 2. Persistent long-term KnowledgeBase (NEW)
        kb = getattr(self.agent.context_manager, "knowledge_base", None)
        if kb and getattr(kb, "enabled", False) and hasattr(kb, "index"):
            for key, meta in kb.index.get("items", {}).items():
                # Create lightweight object compatible with feature extraction
                item = type(
                    "obj",
                    (object,),
                    {
                        "key": key,
                        "content": "",  # content loaded lazily if needed
                        "source_type": "file",
                        "embedding": None,
                        "metadata": meta,
                        "category_path": meta.get("category_path", ""),
                    },
                )()
                candidates.append(item)

        return candidates

    # ====================== FEATURE EXTRACTION ======================
    def _extract_features_sync(
        self,
        query: str,
        candidates: List[Any],
        investigation_state: Dict,
        query_emb: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        rows = []
        query_words = re.findall(r"\b\w+\b", query.lower())

        for item in candidates:
            recency_seconds = time.time() - getattr(item, "timestamp", time.time())
            recency_hours = recency_seconds / 3600.0

            kw = keyword_score(
                query_words=query_words, text=getattr(item, "content", "")
            )
            length = len(getattr(item, "content", ""))
            source_score = self._encode_source_type(getattr(item, "source_type", ""))
            is_tool = 1.0 if getattr(item, "source_type", "") == "tool_outcome" else 0.0

            emb = getattr(item, "embedding", None)
            dense = self._safe_cosine(query_emb, emb) if query_emb is not None else 0.0

            # NEW: Category awareness
            category = getattr(item, "category_path", "") or getattr(
                item, "metadata", {}
            ).get("category_path", "")
            category_score = 1.8 if category else 1.0

            row = {
                "keyword_score": kw,
                "content_length": length,
                "source_type_score": source_score,
                "is_tool_outcome": is_tool,
                "recency_score": np.exp(-0.08 * recency_hours),
                "dense_cosine": dense,
                "cosine_x_keyword": dense * kw * 1.5,
                "semantic_rescue": dense * 2.0 if kw < 4.6 else 0.0,
                "kw_x_length": kw * np.log1p(length) * 1.1,
                "tool_x_source": is_tool * source_score * 2.2,
                "category_score": category_score,  # NEW FEATURE
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if self.active_features:
            df = df.reindex(columns=self.active_features, fill_value=0.0)
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
            "file": 1.8,
        }
        return mapping.get(str(st).lower(), 1.0)

    # ====================== RERANKING ======================
    async def rerank(
        self,
        query: str,
        candidates: Optional[List[Any]] = None,
        k: int = 20,
    ) -> List[Any]:
        if candidates is None:
            candidates = self._get_all_candidates()

        if len(candidates) <= k:
            return candidates[:k]

        fetch_emb = getattr(self.agent.context_manager, "fetch_embedding", None)
        query_emb = await fetch_emb(query, prefix="query") if fetch_emb else None

        features_df = await asyncio.to_thread(
            self._extract_features_sync,
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
            query_emb,
        )

        if self.reranker_model is not None:
            scores_raw = await asyncio.to_thread(
                self.reranker_model.predict, features_df
            )
            scores = np.asarray(scores_raw, dtype=np.float64).flatten()
        else:
            kw = np.array(features_df.get("keyword_score", 0.0))
            length = np.array(features_df.get("content_length", 0.0))
            source = np.array(features_df.get("source_type_score", 0.0))
            tool = np.array(features_df.get("is_tool_outcome", 0.0))
            dense = np.array(features_df.get("dense_cosine", 0.0))
            cat = np.array(features_df.get("category_score", 1.0))

            scores = (
                kw * 0.40
                + np.log1p(length) * 0.18
                + source * 0.12
                + tool * 0.08
                + dense * 0.15
                + cat * 0.07  # NEW: category influence in fallback
            )

        scored_items = list(zip(candidates, scores))
        ranked = sorted(scored_items, key=lambda x: float(x[1]), reverse=True)
        reranked_items = [item for item, _ in ranked[:k]]

        self._collect_training_sample(query, candidates, reranked_items)
        return reranked_items

    def _collect_training_sample(
        self, query: str, candidates: List[Any], chosen_items: List[Any]
    ):
        if len(self.training_data) > 1200:
            return

        features_df = self._extract_features_sync(
            query,
            candidates,
            self.state.__dict__ if hasattr(self.state, "__dict__") else {},
        )

        chosen_ids = {id(item) for item in chosen_items}
        new_samples = []

        for i, item in enumerate(candidates):
            row = features_df.iloc[i].to_dict() if not features_df.empty else {}
            label = 2 if id(item) in chosen_ids else 0
            new_samples.append((row, label))

        self.training_data.extend(new_samples)

        if len(self.training_data) > 900:
            self.training_data = self.training_data[-700:]

        pos = sum(1 for _, label in self.training_data if label > 0)
        logger.debug(
            f"[TRAINING DATA] Collected {len(new_samples)} samples | "
            f"total={len(self.training_data)} | positive={pos}"
        )

        now = time.time()
        if len(self.training_data) >= self.min_samples_for_training:
            if now - getattr(self, "_last_training_ts", 0.0) > 60.0:
                self._last_training_ts = now
                logger.info(
                    f"[TRAINING] Triggering retrain with {len(self.training_data)} samples"
                )
                asyncio.create_task(self._schedule_training())

    # ====================== CONCEPT EXTRACTION ======================
    async def extract_and_consolidate(
        self, trace: Optional[List[Dict]] = None, task_goal: str = ""
    ):
        now = time.time()
        if now - self._last_extraction_ts < self.extraction_cooldown:
            return
        self._last_extraction_ts = now

        if trace is None or len(trace) == 0:
            trace = (
                getattr(self.agent.context_manager, "tool_call_history", [])[-30:]
                + getattr(self.agent.context_manager, "action_log", [])[-30:]
            )

        if not trace:
            return

        goal = getattr(self.agent.state, "task", None) or task_goal or "general task"
        hypotheses = getattr(self.agent.state, "hypotheses", []) or getattr(
            self.agent.context_manager, "investigation_state", {}
        ).get("hypotheses", [])
        findings = getattr(self.agent.state, "findings", []) or getattr(
            self.agent.context_manager, "investigation_state", {}
        ).get("findings", [])

        logger.info(f"[CONSOLIDATE] Starting extraction | goal: {goal[:80]}...")

        # === NEW: Use both short-term and persistent KnowledgeBase ===
        candidates = self._get_all_candidates()
        relevant = await self.rerank(goal, candidates, k=60)

        novel_relevant = self._filter_novel_items(relevant)

        if not novel_relevant:
            logger.debug("[CONSOLIDATE] All candidates already known — skipping")
            return

        extracted = await self._distill_concepts(
            novel_relevant, goal, hypotheses, findings, self.concept_graph
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

        self._update_concept_graph(extracted)

        if added >= 1:
            asyncio.create_task(self._schedule_training())

    # ====================== NOVELTY FILTER (unchanged) ======================
    def _filter_novel_items(self, candidates: List[Any]) -> List[Any]:
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
                description = str(c.get("description") or "").strip()
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

    # ====================== TRAINING ======================
    async def _schedule_training(self):
        current_samples = len(self.training_data)
        if current_samples < self.min_samples_for_training:
            return

        logger.info(f"[TRAINING] Starting retrain with {current_samples} samples")

        try:
            X_list = [sample[0] for sample in self.training_data]
            y = np.array([sample[1] for sample in self.training_data])
            X = pd.DataFrame(X_list)
            groups = [len(X)] if len(X) > 0 else [1]

            await self.train_reranker(X, y, groups)

            if len(self.training_data) > 800:
                self.training_data = self.training_data[-600:]

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
            return

        model_path = self._get_model_path()
        logger.info(f"Training LambdaMART on {len(X)} samples → {model_path}")

        try:
            init_model = None
            if os.path.exists(model_path):
                try:
                    init_model = lgb.Booster(model_file=model_path)
                except Exception:
                    pass

            train_set = lgb.Dataset(X, label=y, group=groups)

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "learning_rate": 0.035,
                "num_leaves": 32,
                "min_data_in_leaf": 4,
                "feature_fraction": 0.88,
                "bagging_fraction": 0.82,
                "bagging_freq": 3,
                "lambda_l1": 0.1,
                "lambda_l2": 0.15,
                "verbose": -1,
            }

            self.reranker_model = lgb.train(
                params,
                train_set,
                num_boost_round=300,
                valid_sets=[train_set],
                init_model=init_model,
                callbacks=[
                    lgb.early_stopping(35, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            self.feature_names = list(X.columns)
            num_trees = self.reranker_model.num_trees()
            importances = dict(
                zip(self.feature_names, self.reranker_model.feature_importance())
            )

            logger.info(
                f"✅ LambdaMART trained | trees={num_trees} | feature_importances: {importances}"
            )
            self.reranker_model.save_model(model_path)

        except Exception as e:
            logger.error(f"Failed to train reranker: {e}", exc_info=True)

    async def load_reranker(self):
        model_path = self._get_model_path()
        try:
            if not os.path.exists(model_path):
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
        logger.info("🔄 Resetting LambdaMART model")
        self.training_data.clear()
        self.reranker_model = None
        self.feature_names = []

        if not keep_graph:
            self.concept_graph.clear()

        model_path = self._get_model_path()
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except Exception:
                pass

        logger.info("✅ Model reset complete.")
