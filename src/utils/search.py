from rapidfuzz import fuzz


def keyword_score(query_words, text):
    words = text.split()

    overlap = len(set(query_words) & set(words))
    coverage = overlap / max(len(query_words), 1)

    prefix_bonus = sum(1 for qw in query_words for w in words if w.startswith(qw))

    return coverage * 3 + prefix_bonus * 0.5


def fuzzy_score(query, text):
    return fuzz.partial_ratio(query, text) / 100
