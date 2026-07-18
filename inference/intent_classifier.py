"""
Rule-based intent classifier for onboarding queries.

Maps natural-language questions to ChromaDB retrieval buckets used in v3:
  code_contracts, wikis_arch_docs, bug_history, community_qa, reference
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(Enum):
    CODE_LOOKUP = "code_contracts"
    API_REFERENCE = "reference"
    BUG_SEARCH = "bug_history"
    COMMUNITY = "community_qa"
    CONCEPT_LOOKUP = "wikis_arch_docs"
    EXAMPLE_REQUEST = "code_contracts"  # prefix intent; searches contracts + docs
    DECISION_LOOKUP = "wikis_arch_docs"
    AMBIGUOUS = "ambiguous"


@dataclass
class ClassifiedQuery:
    original: str
    rewritten: str
    intent: Intent
    confidence: str  # high | medium | low
    search_types: list[str]  # Chroma bucket names for filtering
    retrieval_intent: str  # bucket used for hybrid query prefix


INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.API_REFERENCE, [
        r"\b(parameters?|arguments?|args?|signature|return type|default value)\b",
        r"\bclass [A-Z]\w+\b",
        r"\bHTTPException\b|\bAPIRouter\b|\bDepends\b|\bPath\b|\bQuery\b",
    ]),
    (Intent.CODE_LOOKUP, [
        r"\b(what arguments?|what params?|signature of|return type of)\b",
        r"\b(takes|accepts)\b.{0,30}\b(argument|param|input)\b",
        r"\bhow (is|was) .{0,30} (defined|declared|typed)\b",
        r"\bdef .+\(|class .+:\b",
        r"\b(function|method|class)\b.{0,40}\b(do|accept|return)\b",
    ]),
    (Intent.EXAMPLE_REQUEST, [
        r"\b(show me|give me|example of|sample|snippet|boilerplate|template)\b",
        r"\bhow (do I|to|can I) (write|create|define|build|implement)\b",
        r"\bwhat does.{0,30}look like\b",
    ]),
    (Intent.BUG_SEARCH, [
        r"\b(error|exception|traceback|stacktrace)\b",
        r"\b(not working|doesn'?t work|broken|fails?|crashing)\b",
        r"\b(getting|throws?|raises?)\b.{0,30}\b(error|exception)\b",
    ]),
    (Intent.DECISION_LOOKUP, [
        r"\bwhy (was|were|did|is|are)\b.{0,40}\b(added|introduced|designed|built|changed|removed|deprecated)\b",
        r"\b(reasoning|rationale|motivation|intention) (behind|for|of)\b",
        r"\bwhat (was|is) the (reason|purpose|goal|decision)\b",
        r"\bhow (was|were) .{0,30} (implemented|designed|architected)\b",
    ]),
    (Intent.CONCEPT_LOOKUP, [
        r"\b(what is|what are|explain|describe|tell me about)\b",
        r"\bhow does .{0,30} work\b",
        r"\b(deploy|setup|configure|install|run)\b",
    ]),
]

MEDIUM_PATTERNS: list[tuple[Intent, str]] = [
    (Intent.BUG_SEARCH, r"\b(issue|problem|trouble|debug|fix)\b"),
    (Intent.CODE_LOOKUP, r"\b(function|method|class|import|module)\b"),
    (Intent.CONCEPT_LOOKUP, r"\bhow (to|do)\b"),
    (Intent.COMMUNITY, r"\b(best practice|recommended|should I|vs\.?|difference between)\b"),
    (Intent.DECISION_LOOKUP, r"\bwhy\b"),
]

REWRITE_PREFIXES = {
    Intent.CODE_LOOKUP: "Represent this for searching code signatures and API definitions: ",
    Intent.API_REFERENCE: "Represent this for searching API reference documentation: ",
    Intent.BUG_SEARCH: "Represent this for searching bug reports and issue discussions: ",
    Intent.COMMUNITY: "Represent this for searching Q&A and community discussions: ",
    Intent.CONCEPT_LOOKUP: "Represent this for searching technical documentation: ",
    Intent.EXAMPLE_REQUEST: "Represent this for searching code signatures and API definitions: ",
    Intent.DECISION_LOOKUP: "Represent this for searching technical documentation: ",
    Intent.AMBIGUOUS: "Represent this for searching technical documentation: ",
}

# Each intent lists types most-relevant first. Confidence controls how many
# types are searched: high = primary cluster, medium = broader, low = all.
# Grouped because answers for a "how/what" query often live across types
# (e.g. an API's signature is in `reference`, its real usage is in `code_contracts`).
SEARCH_TYPES_BY_INTENT: dict[Intent, dict[str, list[str]]] = {
    Intent.API_REFERENCE: {
        "high": ["reference", "code_contracts"],
        "medium": ["reference", "code_contracts", "wikis_arch_docs"],
    },
    Intent.CODE_LOOKUP: {
        "high": ["code_contracts", "reference"],
        "medium": ["code_contracts", "reference", "wikis_arch_docs"],
    },
    Intent.EXAMPLE_REQUEST: {
        "high": ["code_contracts", "wikis_arch_docs"],
        "medium": ["code_contracts", "wikis_arch_docs", "community_qa"],
    },
    Intent.BUG_SEARCH: {
        "high": ["bug_history", "community_qa"],
        "medium": ["bug_history", "community_qa", "code_contracts"],
    },
    Intent.DECISION_LOOKUP: {
        "high": ["wikis_arch_docs", "community_qa"],
        "medium": ["wikis_arch_docs", "community_qa", "bug_history"],
    },
    Intent.CONCEPT_LOOKUP: {
        "high": ["wikis_arch_docs", "reference"],
        "medium": ["wikis_arch_docs", "reference", "code_contracts"],
    },
    Intent.COMMUNITY: {
        "high": ["community_qa", "wikis_arch_docs"],
        "medium": ["community_qa", "wikis_arch_docs", "code_contracts"],
    },
}


def classify_intent(query: str) -> ClassifiedQuery:
    q = query.lower().strip()

    for intent, patterns in INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, q):
                return _make_result(query, intent, "high")

    for intent, pattern in MEDIUM_PATTERNS:
        if re.search(pattern, q):
            return _make_result(query, intent, "medium")

    return ClassifiedQuery(
        original=query,
        rewritten=REWRITE_PREFIXES[Intent.AMBIGUOUS] + query,
        intent=Intent.AMBIGUOUS,
        confidence="low",
        search_types=[
            "wikis_arch_docs",
            "code_contracts",
            "bug_history",
            "community_qa",
            "reference",
        ],
        retrieval_intent="wikis_arch_docs",
    )


def _make_result(query: str, intent: Intent, confidence: str) -> ClassifiedQuery:
    search_types = SEARCH_TYPES_BY_INTENT.get(intent, {}).get(
        confidence,
        ["wikis_arch_docs"],
    )
    retrieval_intent = search_types[0]
    if intent == Intent.EXAMPLE_REQUEST:
        retrieval_intent = "code_contracts"
    prefix = REWRITE_PREFIXES.get(intent, REWRITE_PREFIXES[Intent.AMBIGUOUS])
    return ClassifiedQuery(
        original=query,
        rewritten=prefix + query,
        intent=intent,
        confidence=confidence,
        search_types=search_types,
        retrieval_intent=retrieval_intent,
    )
