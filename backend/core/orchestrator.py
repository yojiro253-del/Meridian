from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from backend.core.state import AsyncAgentStateManager
from backend.core.llm import generate_research, decompose_query, quick_search


class AsynchronousOrchestrator:
    """Coordinates multi-step research and streams SSE packets."""

    def __init__(self, session_id, state_manager):
        self.session_id = session_id
        self.state_manager = state_manager

    async def execute_agent_loop(self, prompt):
        """Run real multi-step research and stream each phase as SSE."""
        try:
            # Phase 1: Initialize session
            yield self._sse({
                "phase": "initializing",
                "message": "Starting research session...",
            })

            # Create or get session, save the query
            try:
                session = await self.state_manager.get_session(self.session_id)
                if not session:
                    session = await self.state_manager.new_session(prompt[:80])
                    self.session_id = session["id"]
            except Exception:
                session = await self.state_manager.new_session(prompt[:80])
                self.session_id = session["id"]

            query_record = await self.state_manager.add_query(self.session_id, prompt)
            query_id = query_record["id"]

            # Phase 2: Decompose the query
            yield self._sse({
                "phase": "planning",
                "message": "Decomposing research question into sub-queries...",
            })

            sub_questions = await decompose_query(prompt)

            yield self._sse({
                "phase": "planning",
                "payload": {"sub_questions": sub_questions},
                "message": f"Identified {len(sub_questions)} research angles.",
            })

            # Phase 3: Search each sub-question
            yield self._sse({
                "phase": "searching",
                "message": "Searching scientific literature...",
            })

            search_results = []
            for i, sq in enumerate(sub_questions):
                yield self._sse({
                    "phase": "searching",
                    "message": f"Researching ({i+1}/{len(sub_questions)}): {sq[:60]}...",
                })
                result = await quick_search(sq)
                search_results.append({"question": sq, "findings": result})

            yield self._sse({
                "phase": "searching",
                "payload": {"results_count": len(search_results)},
                "message": f"Gathered findings from {len(search_results)} sub-queries.",
            })

            # Phase 4: Check for related past research
            yield self._sse({
                "phase": "connecting",
                "message": "Searching past research for connections...",
            })

            related = await self.state_manager.find_related(prompt)
            if related:
                yield self._sse({
                    "phase": "connecting",
                    "message": f"Found {len(related)} related findings from past sessions.",
                    "payload": {"related_count": len(related)},
                })

            # Phase 5: Full synthesis via Gemini
            yield self._sse({
                "phase": "synthesizing",
                "message": "Synthesizing research into structured analysis...",
            })

            research = await generate_research(prompt)
            research_data = research.model_dump()

            # Phase 6: Save everything to database
            yield self._sse({
                "phase": "saving",
                "message": "Saving research to knowledge base...",
            })

            # Save research chunks
            await self.state_manager.save_chunk(
                query_id, self.session_id, "synthesis",
                research_data.get("synthesis", ""), embed=True
            )
            for idea in research_data.get("connectedIdeas", []):
                await self.state_manager.save_chunk(
                    query_id, self.session_id, "connection",
                    f"{idea['title']}: {idea['desc']}", embed=True
                )
            for app in research_data.get("applications", []):
                await self.state_manager.save_chunk(
                    query_id, self.session_id, "application",
                    f"{app['title']}: {app['desc']}", embed=True
                )
            for hyp in research_data.get("hypotheses", []):
                await self.state_manager.save_chunk(
                    query_id, self.session_id, "hypothesis",
                    hyp, embed=True
                )

            # Save knowledge graph nodes and edges
            graph_data = research_data.get("graphData", {})
            node_id_map = {}
            for node in graph_data.get("nodes", []):
                try:
                    db_node = await self.state_manager.add_graph_node(
                        label=node["label"],
                        node_type=node.get("type", "concept"),
                        properties={"r": node.get("r", 12)},
                    )
                    node_id_map[node["id"]] = db_node["id"]
                except Exception:
                    pass

            for link in graph_data.get("links", []):
                src = node_id_map.get(link["source"])
                tgt = node_id_map.get(link["target"])
                if src and tgt:
                    try:
                        await self.state_manager.add_graph_edge(
                            source_node_id=src,
                            target_node_id=tgt,
                            relationship="related",
                            session_id=self.session_id,
                        )
                    except Exception:
                        pass

            # Phase 7: Complete
            yield self._sse({
                "phase": "completed",
                "payload": {
                    "session_id": self.session_id,
                    "research": research_data,
                    "related_past_research": [
                        {"content": r["content"], "similarity": r["similarity"]}
                        for r in related
                    ] if related else [],
                },
            })

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield self._sse({
                "phase": "error",
                "message": f"Research failed: {str(exc)}",
            })

    @staticmethod
    def _sse(packet):
        return f"data: {json.dumps(packet)}\n\n"
