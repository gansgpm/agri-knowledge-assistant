"""Guardrails around question answering.

Input guardrails run before retrieval:
  * prompt-injection detection - blocks attempts to override the assistant's rules
  * topic check - a small LLM classification that keeps the assistant on agriculture

Output guardrails run on the model's answer:
  * citation validation - removes citations that point to no retrieved passage
  * citation requirement - withholds answers that cite no source at all
  * coverage warning - flags answers where most statements carry no citation
  * agrochemical safety note - added when an answer discusses pesticides or doses
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

INSUFFICIENT_TOKEN = "INSUFFICIENT_CONTEXT"

# --------------------------------------------------------------------------
# Input guardrails
# --------------------------------------------------------------------------
INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your|the)\b.{0,20}\b(instructions?|rules|prompts?|context|guidelines)\b",
        r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}\b(system|hidden|initial)\s+(prompt|instructions?|message)\b",
        r"\bsystem\s+prompt\b",
        r"\byou\s+are\s+now\b",
        r"\b(pretend|act)\s+(to\s+be|as\s+if|as)\b.{0,40}\b(unrestricted|no\s+rules|without\s+(rules|restrictions)|jailbroken|another\s+(ai|assistant))\b",
        r"\bjailbreak|\bdeveloper\s+mode\b|\bDAN\s+mode\b",
        r"\bdo\s+not\s+(use|follow)\s+the\s+(context|sources|rules)\b",
        r"\banswer\s+(from|using)\s+your\s+own\s+knowledge\b",
    )
]

INJECTION_MESSAGE = (
    "This question was blocked by a safety guardrail because it appears to try to change "
    "the assistant's instructions. Please ask a plain agriculture question."
)
OFF_TOPIC_MESSAGE = (
    "This assistant answers agriculture-related questions only (crops, soil, water, "
    "fertilisers, pests, livestock, farm practices, schemes and markets). "
    "Please rephrase your question for this domain."
)

TOPIC_SYSTEM_PROMPT = """You are a strict classifier for an agriculture assistant.
Decide whether the user's question is related to agriculture in the broad sense:
crops, horticulture, soil, water and irrigation, fertilisers and manure, pests and
diseases, weeds, seeds, livestock, dairy, poultry, fisheries, forestry, sericulture,
farm machinery, weather or climate for farming, post-harvest and food processing,
agricultural statistics, economics, markets, prices, credit, insurance, policies,
government schemes, or rural livelihoods. Questions about agriculture in a named
place (for example a state or district) are related.
Reply with exactly one word: YES or NO."""


def detect_injection(question: str) -> bool:
    """True if the question matches a known prompt-injection pattern."""
    return any(p.search(question) for p in INJECTION_PATTERNS)


class TopicGuard:
    """LLM-based agriculture relevance check. Fails open: if the classifier errors,
    the question is allowed (grounding still prevents made-up answers)."""

    def __init__(self, llm: Runnable):
        prompt = ChatPromptTemplate.from_messages(
            [("system", TOPIC_SYSTEM_PROMPT), ("human", "Question: {question}")]
        )
        self.chain: Runnable = prompt | llm | StrOutputParser()

    def is_on_topic(self, question: str) -> bool:
        try:
            verdict = self.chain.invoke({"question": question}).strip().upper()
        except Exception:
            logger.exception("Topic guardrail failed; allowing the question")
            return True
        on_topic = not verdict.startswith("NO")
        logger.info("Topic guardrail verdict=%r on_topic=%s", verdict[:10], on_topic)
        return on_topic


# --------------------------------------------------------------------------
# Output guardrails
# --------------------------------------------------------------------------
_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_AGROCHEMICAL = re.compile(
    r"\b(pesticides?|insecticides?|fungicides?|herbicides?|weedicides?|nematicides?|"
    r"rodenticides?|acaricides?|agro-?chemicals?|spray(ing|ed)?|ml\s*/\s*l(itre)?|g\s*/\s*l(itre)?|"
    r"a\.i\.|active ingredient)\b",
    re.IGNORECASE,
)
AGROCHEMICAL_NOTE = (
    "Agrochemical advice: always follow the product label, wear protective equipment, "
    "and check with your local agriculture extension officer before spraying."
)
NO_CITATION_WARNING = (
    "The model's answer did not cite any source passage, so it was withheld by the "
    "citation guardrail."
)
LOW_COVERAGE_WARNING = (
    "Some statements in this answer are not linked to a source. Check them against the "
    "citations before acting on them."
)


@dataclass
class CheckedAnswer:
    text: str
    refs: list[int]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    withheld: bool = False


def check_answer(raw: str, num_passages: int, require_citations: bool) -> CheckedAnswer:
    """Validate and clean a model answer against the passages it was given."""
    text = raw.replace(INSUFFICIENT_TOKEN, "").strip()
    warnings: list[str] = []

    # 1. Citations must point to a retrieved passage.
    invalid: set[int] = set()

    def _fix(match: re.Match[str]) -> str:
        kept = []
        for part in match.group(1).split(","):
            ref = int(part)
            if 1 <= ref <= num_passages:
                kept.append(str(ref))
            else:
                invalid.add(ref)
        return f"[{', '.join(kept)}]" if kept else ""

    text = _CITE_GROUP.sub(_fix, text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()
    if invalid:
        listed = ", ".join(f"[{r}]" for r in sorted(invalid))
        warnings.append(f"Removed citation(s) {listed} that pointed to no retrieved passage.")

    refs: list[int] = []
    for group in _CITE_GROUP.findall(text):
        for part in group.split(","):
            ref = int(part)
            if ref not in refs:
                refs.append(ref)

    # 2. An answer must cite at least one passage.
    if not refs:
        if require_citations:
            return CheckedAnswer(text="", refs=[], warnings=[*warnings, NO_CITATION_WARNING], withheld=True)
        warnings.append("The answer cited no passage; all retrieved sources are listed instead.")
        refs = list(range(1, num_passages + 1))

    # 3. Most statements should carry a citation.
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if len(s.strip()) > 25]
    if sentences:
        uncited = sum(1 for s in sentences if not _CITE_GROUP.search(s))
        if uncited / len(sentences) > 0.5:
            warnings.append(LOW_COVERAGE_WARNING)

    # 4. Domain safety note for agrochemicals.
    notes = [AGROCHEMICAL_NOTE] if _AGROCHEMICAL.search(text) else []

    return CheckedAnswer(text=text, refs=refs, warnings=warnings, notes=notes)
