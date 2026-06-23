"""POST /api/research — one-shot JSON research endpoint with database persistence."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Request

from backend.core.llm import generate_research, generate_literature_review
from backend.core.papers import search_papers, get_citation_network
from backend.core.state import AsyncAgentStateManager

router = APIRouter(tags=["Research"])

_STATE_MANAGER = AsyncAgentStateManager()


async def _connect_related_query_node(db_node_id, session_id, query_text):
    related = await _STATE_MANAGER.find_related(query_text, threshold=0.4, limit=20)
    print(f"related chunks: {len(related)}")

    if not related:
        return

    graph_state = await _STATE_MANAGER.get_graph()
    existing_nodes = graph_state.get("nodes", [])

    session_node_map = {}
    for n in existing_nodes:
        props = n.get("properties", {}) or {}
        sid = props.get("sessionId") or props.get("session_id")
        if sid:
            session_node_map[sid] = n

    connected = set()
    for chunk in related:
        chunk_sid = chunk.get("session_id")
        sim = chunk.get("similarity", 0.0)
        if not chunk_sid or chunk_sid == session_id:
            continue
        if chunk_sid in connected:
            continue
        if sim < 0.6:
            continue
        connected.add(chunk_sid)

        related_node = session_node_map.get(chunk_sid)
        if related_node:
            try:
                await _STATE_MANAGER.add_graph_edge(
                    source_node_id=db_node_id,
                    target_node_id=related_node["id"],
                    relationship="related",
                    session_id=session_id,
                    weight=round(sim, 3),
                )
            except Exception as exc:
                print(f"Graph edge save failed: {exc}")


@router.post("/api/research")
async def research(request: Request):
    start = time.time()
    try:
        body = await request.json()
        query_text = body.get("query", "").strip()
        user_instructions = body.get("user_instructions", "")
        mode = body.get("mode", "standard")
        user_id = body.get("user_id", None)

        if not query_text:
            raise HTTPException(status_code=400, detail="query is required")

        state = AsyncAgentStateManager(user_id=user_id)

        # Paper search — same for both modes
        papers_start = time.time()
        papers = await search_papers(query_text)
        print(f"Paper search took {time.time()-papers_start:.1f}s, found {len(papers)} papers")

        gemini_start = time.time()
        if mode == "literature_review":
            result = await generate_literature_review(query_text, user_instructions, papers)
        else:
            result = (await generate_research(query_text, user_instructions, papers)).model_dump()
        print(f"Gemini took {time.time()-gemini_start:.1f}s (elapsed {time.time()-start:.1f}s)")

        # Add top-level metadata
        result["mode"] = mode
        result["papers"] = papers

        # Session management — reuse matching or create new
        session_start = time.time()
        recent = await state.list_sessions()
        session = None
        for s in recent[:5]:
            if s.get("title", "").strip() == query_text[:80].strip():
                session = s
                break
        if session is None:
            session = await state.new_session(query_text[:80])
        query_record = await state.add_query(session["id"], query_text)
        result["sessionId"] = session["id"]
        result["sessionTitle"] = session.get("title", query_text[:80])
        print(f"Session setup took {time.time()-session_start:.1f}s")

        # Find cross-session discoveries
        try:
            discoveries = await state.find_discoveries(query_text, session["id"])
            if discoveries:
                print(f"Found {len(discoveries)} cross-session discoveries")
            result["discoveries"] = discoveries
        except Exception as exc:
            print(f"Discoveries failed: {exc}")
            result["discoveries"] = []

        # Save research chunks to database (concurrent)
        try:
            save_start = time.time()
            synthesis_data = result.get("synthesis", "")
            if isinstance(synthesis_data, list):
                embed_text = " ".join(c.get("claim", "") if isinstance(c, dict) else str(c) for c in synthesis_data)
            else:
                embed_text = synthesis_data

            save_tasks = []

            # Synthesis
            save_tasks.append(state.save_chunk(
                query_record["id"], session["id"], "synthesis",
                embed_text, embed=True,
                metadata={"graph": json.dumps(result.get("graphData", {}) or {}), "papers": json.dumps(papers or []), "synthesis_raw": json.dumps(synthesis_data)},
            ))

            # Connected ideas
            for idea in result.get("connectedIdeas", []):
                save_tasks.append(state.save_chunk(
                    query_record["id"], session["id"], "connection",
                    f"{idea.get('title', '')}: {idea.get('desc', '')}",
                    embed=True,
                    metadata={"id": idea.get("id", ""), "title": idea.get("title", ""), "field": idea.get("field", ""), "desc": idea.get("desc", ""), "confidence": idea.get("confidence", 85)},
                ))

            # Applications
            for app in result.get("applications", []):
                save_tasks.append(state.save_chunk(
                    query_record["id"], session["id"], "application",
                    f"{app.get('title', '')}: {app.get('desc', '')}",
                    embed=True,
                    metadata={"title": app.get("title", ""), "desc": app.get("desc", ""), "impact": app.get("impact", "Medium"), "timeline": app.get("timeline", "Unknown")},
                ))

            # Hypotheses
            for hyp in result.get("hypotheses", []):
                save_tasks.append(state.save_chunk(
                    query_record["id"], session["id"], "hypothesis",
                    hyp if isinstance(hyp, str) else str(hyp), embed=True,
                ))

            await asyncio.gather(*save_tasks)
            print(f"Chunk saving took {time.time()-save_start:.1f}s (elapsed {time.time()-start:.1f}s)")

            # Save a single knowledge-graph node per query
            query_node_label = query_text[:50].strip()
            if query_node_label:
                try:
                    graph_start = time.time()
                    db_node = await state.add_graph_node(
                        label=query_node_label,
                        node_type="concept",
                        properties={
                            "r": 18,
                            "sessionId": session["id"],
                            "sourceQuery": query_text,
                        },
                    )
                    print(f"Graph node saving took {time.time()-graph_start:.1f}s (elapsed {time.time()-start:.1f}s)")

                    try:
                        await asyncio.wait_for(
                            _connect_related_query_node(db_node["id"], session["id"], query_text),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        print("Connection logic timed out after 5s, skipping connections")
                    except Exception as exc:
                        print(f"Connection logic failed: {exc}")
                except Exception as exc:
                    print(f"Graph saving failed: {exc}")
        except Exception as exc:
            print(f"Chunk saving failed: {exc}")

        print(f"Total took {time.time()-start:.1f}s")
        return {"papers": papers, "research": result, "mode": mode}

    except Exception as exc:
        print(f"Research failed after {time.time()-start:.1f}s: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Research generation failed. Please try again.",
        ) from exc


@router.post("/api/preferences")
async def save_preferences(data: dict):
    return {"status": "saved"}


@router.get("/api/papers/network")
async def paper_network(paper_id: str, source: str = "semantic_scholar"):
    """Get citation network for a paper."""
    try:
        network = await get_citation_network(paper_id, source)
        return network
    except Exception as exc:
        return {"nodes": [], "links": [], "error": str(exc)}
