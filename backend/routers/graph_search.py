"""AI-powered knowledge graph search using Gemini."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import google.genai as genai
from google.genai import types as genai_types

router = APIRouter(tags=["graph"])


class SearchRequest(BaseModel):
    q: str
    labels: list[str]


class SearchResponse(BaseModel):
    matches: list[str]


SEARCH_PROMPT = """\
You are a scientific search assistant. Given a search query and a list of \
node labels from a knowledge graph, identify which labels are related to \
the search term. Consider synonyms, related concepts, broader/narrower terms, \
and adjacent fields. Return a JSON object with a "matches" array containing \
only the matching labels (exact strings from the input list, not modified).

Search query: {query}

Node labels: {labels}

Return: {{"matches": ["label1", "label3"]}}
"""


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


@router.post("/api/graph/search", response_model=SearchResponse)
async def graph_search(req: SearchRequest):
    """Use Gemini Flash to find node labels related to the search query."""
    if not req.q or not req.labels:
        return SearchResponse(matches=[])

    if len(req.labels) > 200:
        req.labels = req.labels[:200]

    client = _get_client()
    prompt = SEARCH_PROMPT.format(query=req.q, labels=json.dumps(req.labels))

    config = genai_types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3.5-flash",
            contents=prompt,
            config=config,
        )
        raw = response.candidates[0].content.parts[0].text if response.candidates else None
        if raw is None:
            return SearchResponse(matches=[])
        data = json.loads(raw)
        matches = data.get("matches", [])
        valid = [m for m in matches if m in req.labels]
        return SearchResponse(matches=valid)
    except Exception:
        return SearchResponse(matches=[])
