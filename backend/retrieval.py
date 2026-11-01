# retrieval.py — routes a question to the menu and/or document collections and merges the results.
# Routing is keyword based on purpose: cheap, predictable and easy to test. Both collections are
# always searched; the route only decides how many results each contributes and in what order.

import os
import re
from dataclasses import dataclass
from itertools import zip_longest

from backend.ingest import DOCS_COLLECTION, MENU_COLLECTION, normalize


MENU_TERMS = re.compile(
    r"\b(menus?|dish(es)?|pizzas?|pastas?|desserts?|dolci|gelato|antipast[io]|appetizers?|starters?|salads?|"
    r"insalat[ae]|secondi|contorni|sides?|mains?|main courses?|fish|seafood|meat|chicken|beef|pork|steak|"
    r"bread|focaccia|cocktails?|spritz|beers?|birr[ae]|coffee|caff[eè]|"
    r"espresso|digestiv[oi]|drinks?|zero.proof|prices?|costs?|how much|cheap(est)?|expensive|ingredients?|"
    r"contains?|made with|what'?s in|what is in|comes? (on|with)|toppings?|allerg(en|ens|y|ies|ic)|gluten|celiac|"
    r"coeliac|dairy|lactose|milk|nuts?|peanuts?|eggs?|shellfish|sesame|soy|vegan|vegetarian|spicy|lunch|dinner|"
    r"set menu|specials?|promotions?|aperitivo|happy hour|barolo weeks)\b",
    re.IGNORECASE,
)
DOC_TERMS = re.compile(
    r"\b(polic(y|ies)|procedures?|protocols?|steps?|how (do|should|can) (i|we)|training|sop|uniforms?|"
    r"dress code|appearance|posture|shifts?|clock(ing)? (in|out)|dayforce|benefits?|goodlife|payroll|"
    r"vacation|sick|efap|hr|handbook|onboarding|opening|closing|duties|side ?work|host(ing)?|greet(ing)?|"
    r"seating|clear(ing)? tables?|bread service|beverage service|service standards?|complaints?|clover|"
    r"pos|tip(s|ping)?|tech ?sheets?|utensils?)\b",
    re.IGNORECASE,
)
# Words from dish names that are too generic to signal a menu question on their own.
NAME_STOPWORDS = {"della", "dello", "alla", "allo", "with", "salad", "add", "choose", "glass", "bottle"}


@dataclass
class RetrievalConfig:
    primary_k: int = 5
    secondary_k: int = 2
    both_k: int = 4
    # Off by default: a useful cut-off has to be chosen from evaluation results, not guessed.
    score_threshold: float | None = None

    @classmethod
    def from_env(cls):
        threshold = os.getenv("RETRIEVAL_SCORE_THRESHOLD")
        return cls(
            primary_k=int(os.getenv("RETRIEVAL_K", "5")),
            secondary_k=int(os.getenv("RETRIEVAL_SECONDARY_K", "2")),
            both_k=int(os.getenv("RETRIEVAL_BOTH_K", "4")),
            score_threshold=float(threshold) if threshold else None,
        )


class Retriever:
    def __init__(self, stores, config: RetrievalConfig | None = None):
        self.menu = stores[MENU_COLLECTION]
        self.docs = stores[DOCS_COLLECTION]
        self.config = config or RetrievalConfig.from_env()
        # Dish names ("carbonara", "margherita") mark a menu question even without menu keywords.
        names = self.menu.get(include=["metadatas"])["metadatas"]
        self.name_words = {
            word for md in names for word in normalize(md.get("name", "")).split()
            if len(word) >= 5 and word not in NAME_STOPWORDS
        }

    def route(self, query):
        words = set(normalize(query).split())
        menu = bool(MENU_TERMS.search(query) or "$" in query or words & self.name_words)
        docs = bool(DOC_TERMS.search(query))
        if menu and not docs:
            return "menu"
        if docs and not menu:
            return "documents"
        return "both"

    def retrieve(self, query, route=None, menu_filter=None, docs_filter=None):
        route = route or self.route(query)
        c = self.config
        k_menu, k_docs = {"menu": (c.primary_k, c.secondary_k),
                          "documents": (c.secondary_k, c.primary_k),
                          "both": (c.both_k, c.both_k)}[route]
        menu = self._search(self.menu, query, k_menu, menu_filter)
        docs = self._search(self.docs, query, k_docs, docs_filter)
        # Scores from different collections aren't comparable, so order by route instead of score.
        if route == "menu":
            return menu + docs
        if route == "documents":
            return docs + menu
        return [d for pair in zip_longest(menu, docs) for d in pair if d is not None]

    def _search(self, store, query, k, where):
        if k <= 0:
            return []
        results = []
        for doc, score in store.similarity_search_with_relevance_scores(query, k=k, filter=where):
            if self.config.score_threshold is not None and score < self.config.score_threshold:
                continue
            doc.metadata["score"] = round(score, 4)
            results.append(doc)
        return results
