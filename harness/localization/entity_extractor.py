"""Extract search terms from the task brief (Stage 2.1).

Purely heuristic; the LLM-assisted expansion lives in `code_retriever.expand_queries`.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    # en
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "then", "than",
    "should", "must", "does", "not", "are", "was", "were", "have", "has", "had", "will",
    "can", "but", "all", "any", "its", "our", "your", "their", "which", "where", "while",
    "after", "before", "each", "also", "only", "such", "some", "more", "less", "case",
    "cases", "code", "test", "tests", "fix", "bug", "issue", "error", "expected", "actual",
    "current", "currently", "return", "returns", "value", "values", "function", "method",
    "class", "file", "files", "use", "used", "using", "example", "see", "via", "per",
    # ru
    "это", "как", "что", "для", "при", "или", "если", "его", "она", "они", "оно", "нет",
    "так", "уже", "ещё", "еще", "все", "всё", "быть", "был", "была", "было", "были", "надо",
    "нужно", "должен", "должна", "должно", "должны", "после", "перед", "через", "между",
    "только", "также", "тоже", "который", "которая", "которое", "которые", "сейчас",
    "ошибка", "ошибку", "баг", "исправить", "функция", "функции", "метод", "класс", "файл",
    "код", "тест", "тесты", "возвращает", "возвращать", "значение", "например", "когда",
}

_CODE_SPAN = re.compile(r"`([^`\n]{2,80})`")
_IDENT = re.compile(
    r"\b(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"   # dotted.path
    r"|[A-Za-z]+_[A-Za-z0-9_]+"                              # snake_case
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+"                    # CamelCase
    r")\b"
)
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_-]{2,}")
_SPLIT_IDENT = re.compile(r"[_.\-]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(ident: str) -> list[str]:
    """`NettingPolicy.settle_refund` -> ['netting', 'policy', 'settle', 'refund']."""
    return [p.lower() for p in _SPLIT_IDENT.split(ident) if len(p) >= 2]


def extract_terms(brief: str, *, limit: int = 40) -> list[str]:
    """Return ranked search terms: explicit identifiers first, then plain words."""
    scores: dict[str, float] = {}

    def bump(term: str, weight: float) -> None:
        key = term.strip().strip(".,;:()[]{}\"'")
        if len(key) < 3:
            return
        scores[key] = scores.get(key, 0.0) + weight

    for span in _CODE_SPAN.findall(brief):
        for piece in re.split(r"[\s(),]+", span):
            bump(piece, 5.0)
    for ident in _IDENT.findall(brief):
        bump(ident, 4.0)
        for part in split_identifier(ident):
            if len(part) >= 3:
                bump(part, 1.0)
    for word in _WORD.findall(brief):
        low = word.lower()
        if low in _STOPWORDS:
            continue
        bump(low, 2.0 if word[:1].isupper() else 1.0)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [term for term, _ in ranked[:limit]]
