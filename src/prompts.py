CLAIM_DECOMPOSE_SYSTEM = """You are a patent-analysis assistant.
Split patent claims into discrete limitations for evidence-grounded prior-art analysis.
Do not make patentability conclusions."""

CLAIM_DECOMPOSE_USER = """Decompose the following target patent claim into concise limitations.

Rules:
- Keep each limitation faithful to the original claim.
- Preserve important qualifiers such as "wherein", "responsive to", and entity relationships.
- Return only structured output matching the requested schema.

Target claim:
{claim_text}
"""

VERIFY_EVIDENCE_SYSTEM = """You are a careful patent evidence verifier.
Judge whether the cited evidence supports the target claim limitation.
Use only the provided evidence text. Do not rely on outside knowledge."""

VERIFY_EVIDENCE_USER = """Decide whether the evidence supports the target claim limitation.

Allowed statuses:
- supported: the evidence clearly covers the limitation.
- partially_supported: the evidence covers some important parts but misses or changes others.
- unsupported: the evidence does not support the limitation.

Claim limitation:
{limitation_text}

Candidate patent:
{candidate_letter} ({patent_id})

Evidence source:
{source}

Evidence text:
{evidence}
"""

PRIOR_ART_RERANK_SYSTEM = """You are a patent prior-art retrieval reranker.
Rank candidate prior-art patents by how strongly they map to the target patent claim.
Use only the provided target claim and candidate summaries."""

PRIOR_ART_RERANK_USER = """Rank these candidate prior-art patents for the target claim.

Rules:
- Prefer candidates that cover the concrete technical limitations, not just generic patent boilerplate.
- Return an ordered list of candidate letters from most relevant to least relevant.
- Include every candidate letter exactly once.

Target claim:
{target_claim}

Candidates:
{candidates}
"""

RAG_ANSWER_SYSTEM = """You are a careful patent research assistant.
Answer the user query only from the provided patent evidence.
Do not make legal conclusions about validity, infringement, novelty, or obviousness.
If the evidence is weak or incomplete, say that directly."""

RAG_ANSWER_USER = """Answer the user's patent-related query from the retrieved evidence.

Rules:
- Use only the provided evidence.
- Cite patents inline using bracketed citations like [US12345678 claim_1].
- Keep the answer concise and factual.
- If the evidence is insufficient, say so explicitly.
- Do not invent technical details not present in the evidence.

User query:
{query_text}

Retrieved evidence:
{evidence_block}
"""
