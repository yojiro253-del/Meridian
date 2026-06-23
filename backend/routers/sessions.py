from __future__ import annotations

import html as html_module
import json
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from backend.core.state import AsyncAgentStateManager
from backend.core.database import get_supabase

def esc(text):
    return html_module.escape(str(text)) if text else ""


router = APIRouter(prefix="/api/sessions", tags=["Sessions"])

_STATE_MANAGER = AsyncAgentStateManager()


class CreateSessionRequest(BaseModel):
    title: str = "New Research Session"


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""


@router.get("")
async def list_sessions(user_id: str = None):
    manager = AsyncAgentStateManager(user_id=user_id)
    sessions = await manager.list_sessions()
    return {"sessions": sessions}


@router.post("")
async def create_session(request: Request):
    body = await request.json()
    user_id = body.get("user_id", None)
    title = body.get("title", "New Research Session")
    manager = AsyncAgentStateManager(user_id=user_id)
    session = await manager.new_session(title)
    return session


@router.delete("/reset-all")
async def reset_all(user_id: str = None):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    db = get_supabase()
    db.table("knowledge_edges").delete().eq("user_id", user_id).execute()
    db.table("knowledge_nodes").delete().eq("user_id", user_id).execute()
    db.table("research_chunks").delete().eq("user_id", user_id).execute()
    db.table("queries").delete().in_("session_id",
        [s["id"] for s in db.table("sessions").select("id").eq("user_id", user_id).execute().data or []]
    ).execute()
    db.table("sessions").delete().eq("user_id", user_id).execute()
    return {"status": "user data cleared"}


@router.get("/all/timeline")
async def get_timeline(user_id: str = None):
    """Return all sessions with their queries, ordered by time."""
    manager = AsyncAgentStateManager(user_id=user_id)
    sessions = await manager.list_sessions()

    timeline = []
    for s in sessions:
        try:
            queries = await manager.get_history(s["id"])
        except Exception:
            queries = []
        timeline.append({
            "id": s["id"],
            "title": s.get("title", ""),
            "created_at": s.get("created_at", ""),
            "updated_at": s.get("updated_at", ""),
            "query_count": len(queries),
            "queries": [
                {
                    "text": q.get("query_text", ""),
                    "created_at": q.get("created_at", ""),
                }
                for q in queries
            ],
        })

    return {"timeline": timeline}


@router.get("/{session_id}")
async def get_session(session_id: str, user_id: str = None):
    manager = AsyncAgentStateManager(user_id=user_id)
    session = await manager.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session


@router.get("/{session_id}/history")
async def get_session_history(session_id: str, user_id: str = None):
    manager = AsyncAgentStateManager(user_id=user_id)
    queries = await manager.get_history(session_id)
    return {"queries": queries}


@router.get("/{session_id}/research")
async def get_session_research(session_id: str, user_id: str = None):
    """Reconstruct the full research response for a saved session."""
    manager = AsyncAgentStateManager(user_id=user_id)
    chunks = get_supabase().table("research_chunks") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()

    synthesis = ""
    ideas = []
    apps = []
    hypotheses = []
    papers = []

    for c in chunks.data:
        t = c["chunk_type"]
        content = c["content"]
        md = c.get("metadata") or {}
        if t == "synthesis":
            synthesis += content + "\n"
        elif t == "connection":
            if md.get("title"):
                ideas.append({
                    "id": md.get("id", ""),
                    "title": md["title"],
                    "field": md.get("field", ""),
                    "desc": md.get("desc", content),
                    "confidence": md.get("confidence", 85),
                })
            else:
                parts = content.split(": ", 1)
                title = parts[0] if parts else content
                desc = parts[1] if len(parts) > 1 else ""
                ideas.append({
                    "id": re.sub(r"[^a-z0-9-]", "", re.sub(r"\s+", "-", title.lower()))[:30] or "idea",
                    "title": title,
                    "field": "Connected Field",
                    "desc": desc,
                    "confidence": 85,
                })
        elif t == "application":
            if md.get("title"):
                apps.append({
                    "title": md["title"],
                    "desc": md.get("desc", content),
                    "impact": md.get("impact", "Medium"),
                    "timeline": md.get("timeline", "Unknown"),
                })
            else:
                parts = content.split(": ", 1)
                title = parts[0] if parts else content
                desc = parts[1] if len(parts) > 1 else ""
                apps.append({
                    "title": title,
                    "desc": desc,
                    "impact": "Medium",
                    "timeline": "Unknown",
                })
        elif t == "hypothesis":
            hypotheses.append(content)

    # Check for graph data stored in synthesis chunk's metadata
    graph = None
    for c in chunks.data:
        if c["chunk_type"] == "synthesis":
            md = c.get("metadata") or {}
            raw_papers = md.get("papers")
            if raw_papers and not papers:
                try:
                    papers = json.loads(raw_papers) if isinstance(raw_papers, str) else raw_papers
                except Exception:
                    papers = []
            raw = md.get("graph")
            if raw:
                try:
                    graph = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    pass
            if graph and papers:
                break

    # Fall back to knowledge_nodes/edges
    if not graph:
        graph = await manager.get_graph()
    if not graph:
        graph = {"nodes": [], "links": []}

    if not synthesis:
        synthesis = "<p>No synthesis available for this session.</p>"
    if not ideas:
        ideas = [{"id": "no-data", "title": "No connected ideas found", "field": "", "desc": "", "confidence": 0}]
    if not apps:
        apps = [{"title": "No applications found", "desc": "", "impact": "", "timeline": ""}]
    if not hypotheses:
        hypotheses = ["No hypotheses generated for this session."]
    if not graph.get("nodes"):
        graph["nodes"] = [{"id": "session", "label": "Session", "type": "research", "r": 16}]
    if not graph.get("links"):
        graph["links"] = []

    session = await manager.get_session(session_id)
    return {
        "sessionId": session_id,
        "sessionTitle": (session or {}).get("title", "Research Session"),
        "query": (session or {}).get("title", ""),
        "synthesis": synthesis,
        "connectedIdeas": ideas,
        "applications": apps,
        "hypotheses": hypotheses,
        "graphData": graph,
        "papers": papers,
    }


@router.get("/{session_id}/export", response_class=HTMLResponse)
async def export_session(session_id: str, style: str = "apa", user_id: str = None):
    """Generate a downloadable research brief from a session."""
    manager = AsyncAgentStateManager(user_id=user_id)
    session = await manager.get_session(session_id)
    if not session:
        return HTMLResponse("<h1>Session not found</h1>", status_code=404)

    title = session.get("title", "Research Session")

    queries = await manager.get_history(session_id)

    chunks = get_supabase().table("research_chunks") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()

    synthesis_chunks = [c for c in chunks.data if c["chunk_type"] == "synthesis"]
    connection_chunks = [c for c in chunks.data if c["chunk_type"] == "connection"]
    application_chunks = [c for c in chunks.data if c["chunk_type"] == "application"]
    hypothesis_chunks = [c for c in chunks.data if c["chunk_type"] == "hypothesis"]

    # Formatting helpers
    def format_apa(paper):
        authors = paper.get("authors", [])
        if not authors:
            author_str = "Unknown"
        elif len(authors) == 1:
            parts = authors[0].split()
            author_str = parts[-1] + ", " + ". ".join(p[0] for p in parts[:-1]) + "." if len(parts) > 1 else authors[0]
        elif len(authors) <= 3:
            formatted = []
            for a in authors:
                parts = a.split()
                formatted.append(parts[-1] + ", " + ". ".join(p[0] for p in parts[:-1]) + "." if len(parts) > 1 else a)
            author_str = ", & ".join(formatted) if len(formatted) == 2 else ", ".join(formatted[:-1]) + ", & " + formatted[-1]
        else:
            parts = authors[0].split()
            author_str = (parts[-1] + ", " + ". ".join(p[0] for p in parts[:-1]) + "." if len(parts) > 1 else authors[0]) + " et al."

        year = paper.get("year", "n.d.")
        paper_title = paper.get("title", "Untitled")
        doi = paper.get("doi", "")
        doi_link = f' <a href="https://doi.org/{esc(doi)}">https://doi.org/{esc(doi)}</a>' if doi else ""
        return f'{author_str} ({year}). {paper_title}.{doi_link}'

    def format_nature(paper, num):
        authors = paper.get("authors", [])
        if not authors:
            author_str = "Unknown"
        elif len(authors) == 1:
            parts = authors[0].split()
            author_str = parts[-1] + ", " + ". ".join(p[0] for p in parts[:-1]) + "." if len(parts) > 1 else authors[0]
        else:
            parts = authors[0].split()
            author_str = (parts[-1] + ", " + ". ".join(p[0] for p in parts[:-1]) + "." if len(parts) > 1 else authors[0]) + " et al."

        year = paper.get("year", "n.d.")
        paper_title = paper.get("title", "Untitled")
        doi = paper.get("doi", "")
        doi_link = f' <a href="https://doi.org/{esc(doi)}">https://doi.org/{esc(doi)}</a>' if doi else ""
        return f'{num}. {author_str} {paper_title} ({year}).{doi_link}'

    # Fetch papers for references
    papers = []
    try:
        from backend.core.papers import search_papers
        db = get_supabase()
        queries_result = db.table("queries").select("query_text").eq("session_id", session_id).order("created_at", ascending=True).limit(1).execute()
        if queries_result.data:
            original_query = queries_result.data[0]["query_text"]
            papers = await search_papers(original_query, limit=8)
    except Exception:
        pass

    # Format references
    refs_html = ""
    if papers:
        refs_html = '<h2>References</h2><div class="references">'
        for i, p in enumerate(papers):
            if style == "nature":
                refs_html += f'<p class="ref">{format_nature(p, i+1)}</p>'
            else:
                refs_html += f'<p class="ref">{format_apa(p)}</p>'
        refs_html += '</div>'

    from datetime import date
    today = date.today().strftime("%B %d, %Y")

    formatted = style.upper()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MERIDIAN Research Brief — {esc(title)}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; line-height: 1.7; }}
  h1 {{ font-size: 2rem; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }}
  h2 {{ font-size: 1.4rem; color: #3d3a50; margin-top: 2em; }}
  h3 {{ font-size: 1.1rem; color: #6b6b8d; }}
  .meta {{ font-size: 0.85rem; color: #9694a8; margin-bottom: 2em; }}
  .synthesis {{ background: #f8f7f4; padding: 20px; border-radius: 8px; margin: 1em 0; }}
  .synthesis h4 {{ margin-top: 1em; color: #1a1a2e; }}
  .chunk {{ padding: 12px 16px; margin: 8px 0; border-left: 3px solid #8b5cf6; background: #faf7f2; border-radius: 0 8px 8px 0; }}
  .chunk.connection {{ border-left-color: #3b82f6; }}
  .chunk.application {{ border-left-color: #10b981; }}
  .chunk.hypothesis {{ border-left-color: #f59e0b; }}
  .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 4px; }}
  .label.connection {{ color: #3b82f6; }}
  .label.application {{ color: #10b981; }}
  .label.hypothesis {{ color: #f59e0b; }}
  .references {{ margin-top: 2em; padding-top: 1em; border-top: 2px solid #1a1a2e; }}
  .ref {{ font-size: 0.9rem; line-height: 1.8; margin-bottom: 0.5rem; padding-left: 2em; text-indent: -2em; color: #3d3a50; }}
  .ref a {{ color: #3b82f6; word-break: break-all; }}
  .format-badge {{ display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 12px; border-radius: 100px; background: rgba(59,130,246,0.08); color: #3b82f6; margin-left: 8px; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e0ddd8; font-size: 0.8rem; color: #9694a8; }}
  @media print {{ body {{ margin: 20px; }} .no-print {{ display: none; }} }}
</style>
</head>
<body>
<h1>MERIDIAN Research Brief <span class="format-badge">{formatted} Format</span></h1>
<div class="meta">
  <strong>{esc(title)}</strong><br>
  Generated by MERIDIAN — AI-Assisted Scientific Research Platform<br>
  Date: {today}<br>
  Queries: {len(queries)}
</div>
"""

    if synthesis_chunks:
        html += '<h2>Research Synthesis</h2>\n'
        for c in synthesis_chunks:
            html += f'<div class="synthesis">{esc(c["content"])}</div>\n'

    if connection_chunks:
        html += '<h2>Connected Ideas</h2>\n'
        for c in connection_chunks:
            html += f'<div class="chunk connection"><div class="label connection">Connection</div>{esc(c["content"])}</div>\n'

    if application_chunks:
        html += '<h2>Real-World Applications</h2>\n'
        for c in application_chunks:
            html += f'<div class="chunk application"><div class="label application">Application</div>{esc(c["content"])}</div>\n'

    if hypothesis_chunks:
        html += '<h2>Generated Hypotheses</h2>\n'
        for i, c in enumerate(hypothesis_chunks):
            html += f'<div class="chunk hypothesis"><div class="label hypothesis">Hypothesis {i+1}</div>{esc(c["content"])}</div>\n'

    html += refs_html

    html += """
<footer>
  <p>Generated by MERIDIAN — AI-powered scientific research synthesis.</p>
</footer>
</body>
</html>"""

    return HTMLResponse(html)


@router.get("/graph/all")
async def get_knowledge_graph(user_id: str = None):
    manager = AsyncAgentStateManager(user_id=user_id)
    graph = await manager.get_graph()
    return graph
