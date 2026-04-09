from __future__ import annotations

from dataclasses import dataclass

from src.claim_analysis import rank_candidate_segments
from src.query_planner import TurnPlan


@dataclass(frozen=True)
class QueryEvidenceSnippet:
    patent_id: str
    title: str
    source: str
    evidence: str
    retrieval_score: float
    segment_score: float

    @property
    def citation(self) -> str:
        return f"{self.patent_id} {self.source}"


def gather_query_evidence(query_text: str, ranked, snippets_per_patent: int = 2) -> list[QueryEvidenceSnippet]:
    snippets: list[QueryEvidenceSnippet] = []
    for result in ranked:
        for source, evidence, segment_score in rank_candidate_segments(query_text, result.candidate)[:snippets_per_patent]:
            snippets.append(
                QueryEvidenceSnippet(
                    patent_id=result.patent_id,
                    title=result.title,
                    source=source,
                    evidence=evidence,
                    retrieval_score=float(result.score),
                    segment_score=float(segment_score),
                )
            )
    snippets.sort(key=lambda item: (item.retrieval_score, item.segment_score), reverse=True)
    return snippets


def build_rag_context(snippets: list[QueryEvidenceSnippet], max_snippets: int = 8, max_chars_per_snippet: int = 800) -> str:
    blocks: list[str] = []
    for snippet in snippets[:max_snippets]:
        evidence = snippet.evidence.strip()
        if len(evidence) > max_chars_per_snippet:
            evidence = evidence[: max_chars_per_snippet - 3] + "..."
        blocks.append(
            "\n".join(
                [
                    f"Patent ID: {snippet.patent_id}",
                    f"Title: {snippet.title}",
                    f"Source: {snippet.source}",
                    f"Retrieval score: {snippet.retrieval_score:.3f}",
                    f"Segment score: {snippet.segment_score:.3f}",
                    f"Evidence: {evidence}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def heuristic_rag_answer(
    query_text: str,
    ranked,
    snippets: list[QueryEvidenceSnippet],
    plan: TurnPlan | None = None,
) -> str:
    if not ranked:
        return "I did not retrieve any patents for the query."

    lead = ranked[0]
    intro = "Grounded answer:"
    if plan is not None:
        if plan.intent == "compare_previous_results":
            intro = "Grounded comparison:"
        elif plan.intent == "aspect_filter":
            intro = "Grounded aspect filter:"
        elif plan.intent == "follow_up_on_previous_results":
            intro = "Grounded follow-up answer:"
        elif plan.intent == "combination_exploration":
            intro = "Grounded combination exploration:"

    lines = [
        intro,
        (
            f"The strongest retrieved match for the query is {lead.patent_id}, "
            f"\"{lead.title}.\""
        ),
    ]
    if snippets:
        first = snippets[0]
        lines.append(
            f"The most relevant supporting segment is {first.source} from {first.patent_id}, "
            f"which overlaps with the query terms in the retrieved evidence."
        )

    if len(ranked) > 1:
        second = ranked[1]
        if plan is not None and plan.intent == "compare_previous_results":
            lines.append(
                f"Compared with {second.patent_id}, the lead patent appears more aligned to the current question "
                f"based on the highest-ranked evidence snippets."
            )
        else:
            lines.append(
                "Other retrieved patents appear partially related but should be treated as supporting leads, "
                "not definitive prior art conclusions."
            )

    citations = ", ".join(f"[{snippet.citation}]" for snippet in snippets[:3])
    if citations:
        lines.append(f"Supporting citations: {citations}")
    lines.append(
        "This answer is heuristic unless LLM grounded answering is enabled. It is a technical relevance summary, not legal advice."
    )
    return "\n\n".join(lines)
