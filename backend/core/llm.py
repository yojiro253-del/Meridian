"""Dual-Gemini LLM service: Flash for research, Pro for synthesis."""

from __future__ import annotations

import asyncio
import json
import os

import google.genai as genai
from google.genai import types as genai_types

from backend.schemas.research import ResearchResponse

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


EXPLORE_SYSTEM_PROMPT = """\
You are a scientific research explorer. Given a research question, conduct a deep, unstructured exploration. Find specific mechanisms, key papers, relevant genes/proteins/compounds, surprising connections to adjacent fields, and potential applications. Write freely — this is raw research, not a final report. Be thorough and specific."""

FLASH_SYSTEM_PROMPT = """\
You are MERIDIAN, an advanced scientific research synthesizer inspired by \
paper-qa methodology. When given a research question you must:

1. Synthesize the current state of knowledge as if you had retrieved and \
analyzed the most relevant scientific papers. Write in an authoritative, \
evidence-grounded style. Cite key concepts, mechanisms, genes, proteins, \
or compounds by name. Use HTML formatting: <p>, <h4>, <em>, <code> tags.

2. Identify 4 connected ideas from adjacent or intersecting scientific fields \
that illuminate the query from unexpected angles.

3. Propose 4 real-world applications with practical impact assessments.

4. Generate 3 bold, testable hypotheses that could advance the field.

5. Construct a knowledge graph with 10-14 nodes and 14-18 links showing \
relationships between concepts.

You MUST respond with valid JSON matching this exact schema:
{
  "synthesis": [
    {"claim": "Key finding about the topic", "confidence": "supported"},
    {"claim": "Another important finding", "confidence": "supported"}
  ],
  "connectedIdeas": [
    {"id": "short-slug", "title": "...", "field": "...", "desc": "...", "confidence": 85}
  ],
  "applications": [
    {"title": "...", "desc": "...", "impact": "High|Medium-High|Medium|Low", "timeline": "N – M yrs"}
  ],
  "hypotheses": ["...", "...", "..."],
  "graphData": {
    "nodes": [
      {"id": "slug", "label": "Display Label", "type": "research|connection|application", "r": 12}
    ],
    "links": [
      {"source": "node-id-1", "target": "node-id-2"}
    ]
  }
}

Rules for the knowledge graph:
- The main query topics should be "research" type nodes with r=18-20
- Connected ideas should be "connection" type nodes with r=12-15
- Applications should be "application" type nodes with r=10-12
- Every node must be connected by at least one link
- Node labels can use \\n for line breaks in the graph display
- Node ids must match the connectedIdeas ids where applicable
- All link source/target values must reference valid node ids

Rules for connectedIdeas:
- Exactly 4 items
- confidence is an integer 0-100 representing how strongly the idea connects

Rules for applications:
- Exactly 4 items
- Each needs an impact rating and estimated timeline

Rules for hypotheses:
- Exactly 3 items
- Each should be a detailed, testable scientific hypothesis (2-3 sentences)

Rules for synthesis:
- Each item in the array is one atomic claim
- Claims should be specific, factual statements about the topic
- Keep each claim to 1-2 sentences
- 6-12 claims total covering different aspects of the question
- "confidence" is one of: "supported", "single_source", "contested", "inferred"
- When papers are provided below, each claim MUST include paper_ids referencing the paper index
"""


CITATION_ADDENDUM = """

For the "synthesis" field, return an array of claim objects instead of a single string.
Each claim must cite exactly which paper(s) support it.

Format:
"synthesis": [
  {
    "claim": "Your factual statement here.",
    "paper_ids": [0, 2],
    "confidence": "supported"
  },
  {
    "claim": "Another finding that papers disagree on.",
    "paper_ids": [1, 3],
    "confidence": "contested"
  }
]

Rules:
- Every claim MUST have at least one paper_id referencing the papers array index
- "confidence" is one of: "supported" (multiple papers agree), "single_source" (only one paper), "contested" (papers disagree), "inferred" (logical extension not directly stated)
- Do NOT make claims that no provided paper supports
- Keep claims atomic — one finding per claim

Additionally, analyze the provided papers and include a "literatureAnalysis" field — structured data instead of plain text:

"literatureAnalysis": {
  "consensus": [
    {"finding": "X mechanism is effective", "supporting_papers": [0, 1, 3], "strength": "strong"}
  ],
  "contradictions": [
    {"claim_a": "Drug X improves outcomes", "papers_a": [0], "claim_b": "Drug X shows no effect", "papers_b": [2], "possible_explanation": "Different dosing protocols"}
  ],
  "gaps": [
    {"description": "No studies examine X in population Y", "relevant_papers": [1, 4]}
  ],
  "field_trajectory": "Brief 1-2 sentence summary of where this field is heading"
}

Rules for literatureAnalysis:
- consensus: 2-3 specific findings the papers agree on, each with "strong" or "moderate" strength
- contradictions: 1-2 genuine disagreements between papers (or empty if none), with a possible_explanation for each
- gaps: 2-3 specific gaps in the literature that future research should address
- field_trajectory: replaces the old "trend" field — a single sentence on where the field is heading
- Be specific — reference actual paper numbers in supporting_papers/relevant_papers arrays
- If fewer than 3 papers are available, do your best with what's there

Also include a "whiteSpaces" field — these are UNSTUDIED INTERSECTIONS between the papers provided. These are specific, actionable research opportunities that exist in the gaps between what the papers have studied.

"whiteSpaces": [
  {
    "intersection": "Field A × Field B",
    "description": "One sentence: what has NOT been studied at this intersection",
    "question": "A specific, testable research question that would fill this gap",
    "method": "Brief suggested methodology to investigate this",
    "impact": "High|Medium|Transformative"
  }
]

Rules for whiteSpaces:
- Identify 3-4 genuine gaps — intersections between papers where NO existing work exists
- Each must combine insights from at least 2 of the provided papers
- The research question must be specific and testable, not vague
- The method should be concrete (name a technique, model organism, or dataset)
- "Transformative" impact = would open an entirely new subfield
- Reference the relevant papers by number [1], [2]
- Think like a grant reviewer: would this get funded? If not, make it more specific.
"""


DECOMPOSE_PROMPT = """\
You are a research planning assistant. Given a scientific research question, \
break it into 3-4 specific sub-questions that would need to be answered to \
provide a comprehensive synthesis. Return valid JSON:
{"sub_questions": ["question 1", "question 2", "question 3"]}
"""

LITERATURE_REVIEW_SYSTEM_PROMPT = """You are a scientific literature review assistant. The user wants a structured literature review, not a general research synthesis.

Given the research question and papers provided, produce a structured literature review in JSON format:

{
  "reviewTitle": "A clear, academic title for this literature review",
  "researchQuestion": "The refined research question",
  "scope": {
    "timespan": "e.g. 2018-2025",
    "fieldsRepresented": ["field1", "field2"],
    "totalPapersAnalyzed": 10
  },
  "thematicSections": [
    {
      "theme": "Name of thematic grouping",
      "summary": "2-3 sentence synthesis of this theme across papers",
      "claims": [
        {
          "claim": "Specific factual claim from the literature",
          "paper_ids": [0, 2],
          "confidence": "supported"
        }
      ],
      "evolution": "How thinking on this theme has changed over time"
    }
  ],
  "methodologicalLandscape": {
    "dominantMethods": ["method1", "method2"],
    "emergingApproaches": ["approach1"],
    "methodologicalGaps": ["gap1"]
  },
  "consensus": [
    {"finding": "What most papers agree on", "supporting_papers": [0, 1, 3], "strength": "strong"}
  ],
  "contradictions": [
    {"claim_a": "One position", "papers_a": [0], "claim_b": "Opposing position", "papers_b": [2], "possible_explanation": "Why they differ"}
  ],
  "gaps": [
    {"description": "What has NOT been studied", "relevant_papers": [1, 4], "importance": "high"}
  ],
  "futureDirections": [
    {"direction": "Suggested research direction", "rationale": "Why this matters", "building_on": [0, 3]}
  ],
  "graphData": {
    "nodes": [{"id": "theme_name", "type": "theme"}],
    "edges": [{"source": "theme1", "target": "theme2", "relationship": "builds_on"}]
  }
}

Rules:
- Every claim MUST cite paper_ids referencing the papers array index
- Organize papers into themes — do NOT just summarize each paper individually
- "confidence" is one of: "supported", "single_source", "contested", "inferred"
- Identify genuine contradictions, not just different topics
- "importance" for gaps is one of: "high", "medium", "low"
- Keep thematic sections to 3-5 themes maximum
"""


async def explore_query(query: str) -> str:
    """Step 1: Deep exploration using Gemini Flash."""
    client = _get_client()

    config = genai_types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=8192,
        system_instruction=EXPLORE_SYSTEM_PROMPT,
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=query,
        config=config,
    )

    raw = response.candidates[0].content.parts[0].text if response.candidates else None
    if raw is None:
        raise RuntimeError("Gemini returned an empty response for exploration.")
    return raw


def format_papers_for_prompt(papers):
    """Build a numbered paper listing for Gemini prompt context."""
    if not papers:
        return ""
    lines = []
    for i, p in enumerate(papers):
        lines.append(f"[{i}] {p.get('title', '')}")
        authors = p.get("authors", [])
        if authors:
            lines.append(f"    Authors: {', '.join(authors[:3])}")
        year = p.get("year")
        if year:
            lines[-1] += f" ({year})"
        abstract = p.get("abstract")
        if abstract:
            lines.append(f"    Abstract: {abstract[:250]}")
        doi = p.get("doi")
        if doi:
            lines.append(f"    DOI: {doi}")
    return "\n".join(lines)


async def generate_research(query: str, user_instructions: str = "", papers=None):
    """Structured synthesis using Gemini Flash."""
    client = _get_client()

    system = FLASH_SYSTEM_PROMPT
    if papers:
        system = system + CITATION_ADDENDUM
    if user_instructions:
        system = (
            "MANDATORY USER INSTRUCTIONS — THESE OVERRIDE ALL OTHER FORMATTING PREFERENCES:\n"
            "You MUST follow every instruction below exactly. Do not deviate, skip, or partially comply.\n"
            "---\n"
            + user_instructions + "\n"
            "---\n"
            "Violation of any user instruction above is a critical error.\n\n"
            + system
        )

    config = genai_types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=8192,
        response_mime_type="application/json",
        system_instruction=system,
    )

    user_content = query
    if papers:
        paper_context = "\n--- REAL SCIENTIFIC PAPERS (cite by number in your synthesis) ---\n"
        paper_context += format_papers_for_prompt(papers)
        user_content = paper_context + "\n--- RESEARCH QUESTION ---\n" + query

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=user_content,
        config=config,
    )

    raw = response.candidates[0].content.parts[0].text if response.candidates else None
    if raw is None:
        raise RuntimeError("Gemini returned an empty response for synthesis.")

    data = json.loads(raw)
    if "connectedIdeas" not in data and "connected_ideas" in data:
        data["connectedIdeas"] = data.pop("connected_ideas")
    if "graphData" not in data and "graph" in data:
        data["graphData"] = data.pop("graph")
    # Normalize graph edges -> links (Gemini sometimes returns "edges")
    gd = data.get("graphData")
    if isinstance(gd, dict):
        if "edges" in gd and "links" not in gd:
            gd["links"] = gd.pop("edges")
    return ResearchResponse.model_validate(data)


async def decompose_query(query):
    """Break a research question into sub-questions using Gemini Flash."""
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=1024,
        response_mime_type="application/json",
        system_instruction=DECOMPOSE_PROMPT,
    )
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=query,
        config=config,
    )
    raw = response.candidates[0].content.parts[0].text if response.candidates else None
    if raw is None:
        return [query]
    data = json.loads(raw)
    return data.get("sub_questions", [query])


async def quick_search(question):
    """Use Gemini Flash to gather information on a specific sub-question."""
    client = _get_client()
    config = genai_types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=2048,
        system_instruction="You are a scientific research assistant. Answer the question with specific facts, mechanisms, and findings. Be concise but thorough.",
    )
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=question,
        config=config,
    )
    raw = response.candidates[0].content.parts[0].text if response.candidates else None
    return raw or ""


async def generate_literature_review(query: str, user_instructions: str = "", papers=None):
    """Structured literature review using Gemini Flash."""
    client = _get_client()

    system = LITERATURE_REVIEW_SYSTEM_PROMPT
    if user_instructions:
        system = (
            "MANDATORY USER INSTRUCTIONS — THESE OVERRIDE ALL OTHER FORMATTING PREFERENCES:\n"
            "You MUST follow every instruction below exactly. Do not deviate, skip, or partially comply.\n"
            "---\n"
            + user_instructions + "\n"
            "---\n"
            "Violation of any user instruction above is a critical error.\n\n"
            + system
        )

    config = genai_types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=8192,
        response_mime_type="application/json",
        system_instruction=system,
    )

    paper_context = format_papers_for_prompt(papers) if papers else ""
    user_prompt = f"""Research question: {query}

Papers found:
{paper_context}

Produce a structured literature review as specified."""

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-3.5-flash",
        contents=user_prompt,
        config=config,
    )

    raw = response.candidates[0].content.parts[0].text if response.candidates else None
    if raw is None:
        raise RuntimeError("Gemini returned an empty response for literature review.")
    return json.loads(raw)
