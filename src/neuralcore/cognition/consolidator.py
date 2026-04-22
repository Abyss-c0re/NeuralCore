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

        # ====================== CONFIG ======================
        app_config = getattr(getattr(self.agent, "loader", None), "config", {}).get(
            "app", {}
        )
        cognition_config = app_config.get("cognition", {})

        self.reranker_enabled: bool = cognition_config.get("reranker_enabled", True)
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
            "category_score",
        ]

        if self.reranker_enabled:
            logger.info(
                f"✅ KnowledgeConsolidator initialized | "
                f"reranker=ENABLED (lazy load) | novelty_threshold={self.novelty_threshold}"
            )
        else:
            logger.info(
                "⚠️  KnowledgeConsolidator initialized | reranker=DISABLED (hybrid only)"
            )

    # ====================== Get candidates from both sources ======================
    def _get_all_candidates(self) -> List[Any]:
        """Combine short-term memory + persistent KnowledgeBase items."""
        candidates: List[Any] = []

        cm = self.agent.context_manager
        if hasattr(cm, "short_term_mem") and isinstance(cm.short_term_mem, dict):
            candidates.extend(list(cm.short_term_mem.values()))

        kb = getattr(cm, "knowledge_base", None)
        if kb and getattr(kb, "enabled", False) and hasattr(kb, "index"):
            for key, meta in kb.index.get("items", {}).items():
                item = type(
                    "KBItemProxy",
                    (object,),
                    {
                        "key": key,
                        "content": "",
                        "source_type": "file",
                        "embedding": None,
                        "metadata": meta,
                        "category_path": meta.get("category_path", ""),
                        "timestamp": meta.get("added_at", time.time()),
                    },
                )()
                candidates.append(item)

        logger.debug(
            f"[CONSOLIDATE] Loaded {len(candidates)} total candidates "
            f"(short-term + persistent)"
        )
        return candidates

    # ====================== FEATURE EXTRACTION ======================
    async def _extract_features(
        self,
        query: str,
        candidates: List[Any],
        query_emb: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Fully async feature extraction with proper embedding."""
        rows = []
        query_words = re.findall(r"\b\w+\b", query.lower())

        # === 2. Fetch query embedding if not provided ===
        if query_emb is None:
            try:
                query_emb = await self.agent.context_manager.fetch_embedding(
                    query, prefix="query"
                )
            except Exception as e:
                logger.warning(f"Failed to embed query: {e}")
                query_emb = None

        # === 3. Extract features for each candidate ===
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
                "category_score": category_score,
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
            "tool": 2.2,
            "agent_trace": 2.5,
            "raw_knowledge": 1.0,
            "file": 1.8,
        }
        return mapping.get(str(st).lower(), 1.0)

    async def _distill_concepts(
        self,
        relevant_items: List[Any],
        goal: str,
        existing_concepts: Optional[Dict[str, Any]] = None,
        max_concepts: int = 7,
        temperature: float = 0.28,
    ) -> List[Dict]:
        if not relevant_items:
            logger.debug("[DISTILL] No relevant items — skipping")
            return []

        prompt = PromptBuilder.abstract_concept_extraction(
            goal=goal,
            relevant_items=relevant_items,
            existing_concepts=existing_concepts,
            max_concepts=max_concepts,
        )

        try:
            response = await self.agent.client.ask(
                prompt, temperature=temperature, max_tokens=5000
            )

            # Robust JSON extraction
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE
            )
            cleaned = cleaned.strip()

            # Try to find JSON array even if there's extra text
            json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            concepts = json.loads(cleaned)

            if isinstance(concepts, list):
                # Filter out low-quality concepts
                valid_concepts = [
                    c
                    for c in concepts
                    if c.get("name")
                    and c.get("description")
                    and c.get("score", 0) > 0.4
                ]

                logger.info(
                    f"[DISTILL] Successfully extracted {len(valid_concepts)} abstract concepts "
                    f"(requested {max_concepts})"
                )
                return valid_concepts[:max_concepts]

        except json.JSONDecodeError as e:
            logger.warning(f"[DISTILL] JSON parsing failed: {e}")
        except Exception as e:
            logger.error(f"[DISTILL] LLM extraction failed: {e}")

        return []

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

        if self.reranker_enabled and self.reranker_model is None:
            await self.load_reranker()

        features_df = await self._extract_features(query, candidates)

        # ====================== ML RERANKING (if enabled & model loaded) ======================
        if self.reranker_enabled and self.reranker_model is not None:
            try:
                scores_raw = await asyncio.to_thread(
                    self.reranker_model.predict, features_df
                )
                scores = np.asarray(scores_raw, dtype=np.float64).flatten()
                logger.debug("[RERANK] Using LambdaMART model")
            except Exception as e:
                logger.warning(
                    f"[RERANK] Model prediction failed: {e} → falling back to hybrid"
                )
                scores = self._hybrid_score(features_df)
        else:
            scores = self._hybrid_score(features_df)

        scored_items = list(zip(candidates, scores))
        ranked = sorted(scored_items, key=lambda x: float(x[1]), reverse=True)
        reranked_items = [item for item, _ in ranked[:k]]

        await self._collect_training_sample(query, candidates, reranked_items)
        return reranked_items

    def _hybrid_score(self, features_df: pd.DataFrame) -> np.ndarray:
        kw = np.array(features_df.get("keyword_score", 0.0))
        length = np.array(features_df.get("content_length", 0.0))
        source = np.array(features_df.get("source_type_score", 0.0))
        tool = np.array(features_df.get("is_tool_outcome", 0.0))
        dense = np.array(features_df.get("dense_cosine", 0.0))
        cat = np.array(features_df.get("category_score", 1.0))

        return (
            kw * 0.35
            + np.log1p(length) * 0.15
            + source * 0.10
            + tool * 0.08
            + dense * 0.15
            + cat * 0.07
        )

    async def _collect_training_sample(
        self, query: str, candidates: List[Any], chosen_items: List[Any]
    ):
        if not getattr(self, "reranker_enabled", True):
            return

        if len(self.training_data) > 1200:
            return

        features_df = await self._extract_features(query, candidates)

        chosen_ids = {id(item) for item in chosen_items}
        new_samples = []

        for i, item in enumerate(candidates):
            row = features_df.iloc[i].to_dict() if not features_df.empty else {}
            label = 2 if id(item) in chosen_ids else 0
            new_samples.append((row, label))

        self.training_data.extend(new_samples)

        # Keep training buffer size under control
        if len(self.training_data) > 900:
            self.training_data = self.training_data[-700:]

        # Auto-trigger training
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

        cm = self.agent.context_manager

        goal = getattr(self.agent.state, "task", None) or task_goal or "general task"

        if not trace or len(trace) == 0:
            trace = (
                getattr(cm, "tool_call_history", [])[-25:]
                + getattr(cm, "action_log", [])[-25:]
            )
        if not trace:
            return

        novel_relevant = await self._get_novel_relevant(goal)

        if not novel_relevant:
            return

        # Abstract concepts
        extracted = await self._distill_concepts(
            novel_relevant, goal, self.concept_graph
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
            if hasattr(cm, "fetch_embedding"):
                item.embedding = await cm.fetch_embedding(
                    concept.get("description", ""), prefix="passage"
                )
            cm.short_term_mem[item.key] = item
            added += 1

        if added > 0:
            if hasattr(cm, "_sparse_index_dirty"):
                cm._sparse_index_dirty = True
            logger.info(f"✅ Stored {added} new abstract concepts")

        self._update_concept_graph(extracted)

        if added >= 1:
            asyncio.create_task(self._schedule_training())

    # ====================== LLM SUGGESTION TRAINING SIGNAL ======================
    async def record_llm_suggested_tool(self, query: str, tool_name: str):
        """Record high-quality training signal from LLM planning stage."""
        if not tool_name or not query or len(self.training_data) > 950:
            return

        if not getattr(self, "reranker_enabled", True):
            return

        # Prevent flooding with too many synthetic tool samples
        existing_tool_samples = sum(
            1 for row, _ in self.training_data if row.get("tool_x_source", 0) > 0.5
        )
        if existing_tool_samples > 140:
            return

        tool_item = type(
            "LLMToolSuggestion",
            (object,),
            {
                "key": tool_name,
                "content": f"LLM recommended for: {query}",
                "source_type": "tool",
                "embedding": None,
                "timestamp": time.time(),
                "usage_count": 1,
                "category_path": "llm_suggested",
            },
        )()

        features_df = await self._extract_features(query, [tool_item])

        if not features_df.empty:
            row = features_df.iloc[0].to_dict()
            self.training_data.append((row, 2))

            logger.info(
                f"[LLM SUGGESTION] Recorded positive sample | tool='{tool_name}' | query='{query[:55]}...'"
            )

            # Auto-trigger training
            if len(self.training_data) >= self.min_samples_for_training:
                now = time.time()
                if now - getattr(self, "_last_training_ts", 0.0) > 45.0:
                    self._last_training_ts = now
                    asyncio.create_task(self._schedule_training())

    async def record_actual_tool_usage(
        self, query: str, tool_name: str, success: bool = True
    ):
        """Even stronger signal — tool was actually executed successfully."""
        if not tool_name or not query or len(self.training_data) > 950:
            return

        if not getattr(self, "reranker_enabled", True):
            return

        existing_tool_samples = sum(
            1 for row, _ in self.training_data if row.get("tool_x_source", 0) > 0.5
        )
        if existing_tool_samples > 160:
            return

        tool_item = type(
            "ActualToolUsage",
            (object,),
            {
                "key": tool_name,
                "content": f"Actually used for: {query}",
                "source_type": "tool_outcome",
                "embedding": None,
                "timestamp": time.time(),
                "usage_count": 5,
                "category_path": "actual_usage",
            },
        )()

        features_df = await self._extract_features(query, [tool_item])

        if not features_df.empty:
            row = features_df.iloc[0].to_dict()
            label = 2 if success else 0
            self.training_data.append((row, label))

            logger.info(
                f"[ACTUAL USAGE] Recorded {'positive' if success else 'negative'} sample | tool='{tool_name}'"
            )

            # Auto-trigger training
            if len(self.training_data) >= self.min_samples_for_training:
                now = time.time()
                if now - getattr(self, "_last_training_ts", 0.0) > 45.0:
                    self._last_training_ts = now
                    asyncio.create_task(self._schedule_training())

    # ====================== NOVELTY FILTER ======================
    async def _get_novel_relevant(self, goal: str):
        candidates = self._get_all_candidates()
        relevant = await self.rerank(goal, candidates, k=50)
        return self._filter_novel_items(relevant)

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

    # ====================== UPDATE CONCEPT GRAPH (FIXED) ======================
    def _update_concept_graph(self, concepts: List[Dict]):
        for c in concepts:
            name = c.get("name")
            if not name:
                continue

            description = str(c.get("description") or "").strip()
            matching_item = None

            # Safe lookup in short-term memory only (avoid .values() on KnowledgeBase object)
            cm = self.agent.context_manager
            if hasattr(cm, "short_term_mem"):
                for item in cm.short_term_mem.values():
                    if (
                        getattr(item, "key", "").startswith("concept_")
                        and description in str(getattr(item, "content", "")).strip()
                    ):
                        matching_item = item
                        break

            if matching_item and getattr(matching_item, "embedding", None) is not None:
                c["embedding"] = matching_item.embedding

            self.concept_graph[name] = c

    # ====================== TRAINING ======================
    async def _schedule_training(self):
        if not self.reranker_enabled:
            return

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
        if not self.reranker_enabled or len(X) < 10:
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
                "min_data_in_leaf": 8,
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
        if not self.reranker_enabled:
            self.reranker_model = None
            return

        model_path = self._get_model_path()
        try:
            if not os.path.exists(model_path):
                logger.info(
                    f"No trained model at {model_path}. Will use hybrid scoring."
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
        if not self.reranker_enabled:
            return

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
