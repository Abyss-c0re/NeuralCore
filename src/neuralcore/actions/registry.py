import math
from typing import Dict
from src.actions.actions import Action, ActionSet

from src.utils.search import keyword_score, fuzzy_score


class ActionRegistry:
    """Central searchable registry of all tools."""

    def __init__(self):
        self.sets: Dict[str, ActionSet] = {}
        self.all_actions: Dict[str, tuple[Action, str]] = {}

        self._index = []  # SEARCH INDEX

    def register_set(self, name: str, action_set: ActionSet):

        if name in self.sets:
            raise ValueError(f"Set {name} already exists")

        self.sets[name] = action_set

        for action in action_set.actions:
            if action.name in self.all_actions:
                print(f"Warning: duplicate tool name {action.name}")

            self.all_actions[action.name] = (action, name)

            # BUILD SEARCH INDEX ENTRY
            searchable = action._search_text

            self._index.append(
                {
                    "action": action,
                    "set": name,
                    "text": searchable,
                }
            )

    def search(self, query: str, limit: int = 8):

        query = query.lower().strip()
        query_words = query.split()

        results = []

        for entry in self._index:
            action = entry["action"]
            text = entry["text"]
            set_name = entry["set"]

            # scoring
            k_score = keyword_score(query_words, text)
            f_score = fuzzy_score(query, text)

            usage_bonus = math.log1p(action.usage_count)

            set_bonus = 0
            if query in set_name.lower():
                set_bonus = 1.5

            total = k_score + f_score * 2 + usage_bonus * 0.5 + set_bonus

            if total > 0.4:
                results.append((total, action, set_name))

        results.sort(key=lambda x: x[0], reverse=True)

        return [(a, s) for _, a, s in results[:limit]]
