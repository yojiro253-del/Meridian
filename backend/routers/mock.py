"""POST /api/research/mock — returns realistic mock data for frontend testing."""

from __future__ import annotations

import json

from fastapi import APIRouter

from backend.core.state import AsyncAgentStateManager
from backend.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter(tags=["Mock"])

_STATE_MANAGER = AsyncAgentStateManager()


@router.post("/api/research/mock", response_model=ResearchResponse)
async def mock_research(request: ResearchRequest = None):
    """Return a realistic structured research payload without calling Gemini."""
    query_text_input = request.query.strip() if request else "MERIDIAN framework analysis [mock data]"

    synthesis_data = (
        "<p>The <strong>MERIDIAN framework</strong> represents a novel approach to open-ended scientific exploration "
        "that combines structured reasoning with associative knowledge mapping. By integrating large language models "
        "with graph-based representation, it enables researchers to traverse hypothesis spaces more efficiently "
        "than traditional linear methods.</p>"
        "<p>Our analysis identifies three core mechanisms underlying this framework: (1) <strong>divergent exploration</strong> "
        "that generates a broad spectrum of candidate connections, (2) <strong>convergent synthesis</strong> that distills "
        "these into coherent knowledge structures, and (3) <strong>recursive refinement</strong> where each cycle deepens "
        "the fidelity of the emerging graph.</p>"
        "<p>Key findings suggest that graph density correlates positively with novelty of generated hypotheses, "
        "while node centrality measures predict the likelihood of cross-disciplinary application discovery. "
        "The approach shows particular promise in domains where established knowledge is fragmented across "
        "multiple sub-disciplines.</p>"
    )

    ideas_data = [
        {
            "id": "idea-1",
            "title": "Graph-Enhanced Hypothesis Generation",
            "field": "Computational Epistemology",
            "desc": "Using knowledge graph topology to seed novel hypothesis spaces by identifying structural gaps and boundary nodes where information flow is constrained. Graph-theoretic measures predict fertile ground for new conjectures.",
            "confidence": 92,
        },
        {
            "id": "idea-2",
            "title": "Recursive Abductive Reasoning Loops",
            "field": "Cognitive Science",
            "desc": "MERIDIAN's recursive refinement mirrors human abductive reasoning, where each iterative pass generates explanatory hypotheses that feed back into the reasoning graph. This models how scientists naturally refine theories.",
            "confidence": 87,
        },
        {
            "id": "idea-3",
            "title": "Cross-Domain Analogy Transfer via Graph Isomorphism",
            "field": "Analogical Reasoning",
            "desc": "Subgraph isomorphism between knowledge domains enables automated transfer of structural analogies, allowing insights from one field to seed discoveries in another through shared relational patterns.",
            "confidence": 78,
        },
        {
            "id": "idea-4",
            "title": "Emergent Property Detection Through Graph Entropy",
            "field": "Complex Systems",
            "desc": "Monitoring entropy changes in the knowledge graph during exploration reveals emergent properties — sudden drops in entropy correlate with pivotal conceptual breakthroughs that restructure the problem space.",
            "confidence": 83,
        },
        {
            "id": "idea-5",
            "title": "Adversarial Knowledge Fragmentation Defense",
            "field": "AI Safety",
            "desc": "Graph-based knowledge representation inherently resists adversarial fragmentation by maintaining multiple redundant pathways between concepts, ensuring robustness against targeted information corruption.",
            "confidence": 71,
        },
    ]

    apps_data = [
        {
            "title": "Drug Discovery Target Identification",
            "desc": "Deploy MERIDIAN to map protein interaction networks and identify novel druggable targets by finding high-centrality nodes at the intersection of disease pathways and existing compound databases.",
            "impact": "Very High",
            "timeline": "3–5 years",
        },
        {
            "title": "Climate Model Intercomparison",
            "desc": "Use the framework to synthesize findings across 40+ climate models, constructing a meta-graph that reveals consensus predictions and structural uncertainties in regional climate projections.",
            "impact": "High",
            "timeline": "1–2 years",
        },
        {
            "title": "Scientific Literature Meta-Review",
            "desc": "Automatically generate living meta-reviews by ingesting new publications into the knowledge graph, updating synthesis and highlighting evolving consensus or emerging contradictions in real time.",
            "impact": "Medium",
            "timeline": "Ongoing",
        },
        {
            "title": "Educational Curriculum Design",
            "desc": "Map prerequisite relationships between concepts across STEM disciplines to design optimized learning pathways that minimize cognitive load while maximizing conceptual coverage.",
            "impact": "Medium",
            "timeline": "6–12 months",
        },
    ]

    hypotheses_data = [
        "Knowledge graph edge density in MERIDIAN follows a power-law distribution where ~20% of nodes account for ~80% of connections, mirroring the Pareto principle in scientific citation networks and suggesting an underlying universality in knowledge structuring.",
        "The iterative refinement loop exhibits diminishing returns beyond 4–5 cycles, with each additional pass contributing less than 5% new information, implying an optimal stopping criterion for automated exploration.",
        "Cross-domain connections (edges linking nodes from different sub-disciplines) have a ~3x higher probability of generating patentable applications compared to intra-domain connections, based on preliminary analysis of analogy transfer success rates.",
    ]

    graph_data = {
        "nodes": [
            {"id": "meridian", "label": "MERIDIAN\nFramework", "type": "research", "r": 22},
            {"id": "hyp-gen", "label": "Hypothesis\nGeneration", "type": "research", "r": 17},
            {"id": "graph-topo", "label": "Graph\nTopology", "type": "concept", "r": 14},
            {"id": "abductive", "label": "Abductive\nReasoning", "type": "concept", "r": 14},
            {"id": "analogy", "label": "Analogical\nTransfer", "type": "connection", "r": 13},
            {"id": "entropy", "label": "Graph\nEntropy", "type": "concept", "r": 13},
            {"id": "adversarial", "label": "Adversarial\nRobustness", "type": "concept", "r": 13},
            {"id": "drug-disc", "label": "Drug\nDiscovery", "type": "application", "r": 16},
            {"id": "climate", "label": "Climate\nModeling", "type": "application", "r": 16},
            {"id": "education", "label": "Education\nDesign", "type": "application", "r": 14},
        ],
        "links": [
            {"source": "meridian", "target": "hyp-gen"},
            {"source": "meridian", "target": "graph-topo"},
            {"source": "meridian", "target": "abductive"},
            {"source": "meridian", "target": "analogy"},
            {"source": "meridian", "target": "entropy"},
            {"source": "meridian", "target": "adversarial"},
            {"source": "hyp-gen", "target": "graph-topo"},
            {"source": "hyp-gen", "target": "entropy"},
            {"source": "abductive", "target": "analogy"},
            {"source": "graph-topo", "target": "entropy"},
            {"source": "drug-disc", "target": "meridian"},
            {"source": "climate", "target": "meridian"},
            {"source": "education", "target": "meridian"},
            {"source": "drug-disc", "target": "hyp-gen"},
            {"source": "climate", "target": "entropy"},
        ],
    }

    mock_data = {
        "synthesis": synthesis_data,
        "connectedIdeas": ideas_data,
        "applications": apps_data,
        "hypotheses": hypotheses_data,
        "graphData": graph_data,
    }

    session = None
    try:
        query_text = query_text_input
        session = await _STATE_MANAGER.new_session(query_text[:80])
        query_record = await _STATE_MANAGER.add_query(session["id"], query_text)

        await _STATE_MANAGER.save_chunk(
            query_record["id"], session["id"], "synthesis",
            synthesis_data, embed=True,
            metadata={"graph": json.dumps(graph_data)},
        )

        for idea in ideas_data:
            await _STATE_MANAGER.save_chunk(
                query_record["id"], session["id"], "connection",
                f"{idea['title']}: {idea['desc']}",
                embed=True,
                metadata=idea,
            )

        for app in apps_data:
            await _STATE_MANAGER.save_chunk(
                query_record["id"], session["id"], "application",
                f"{app['title']}: {app['desc']}",
                embed=True,
                metadata=app,
            )

        for hyp in hypotheses_data:
            await _STATE_MANAGER.save_chunk(
                query_record["id"], session["id"], "hypothesis",
                hyp, embed=True,
            )

        # Save a small chunk with the query text itself for embedding matching
        try:
            await _STATE_MANAGER.save_chunk(
                query_record["id"], session["id"], "synthesis",
                query_text, embed=True,
                metadata={"graph": json.dumps(graph_data)},
            )
        except Exception:
            pass

        # Single query node (not sub-concept nodes)
        try:
            db_node = await _STATE_MANAGER.add_graph_node(
                label=query_text[:50].strip(),
                node_type="concept",
                properties={
                    "r": 18,
                    "sessionId": session["id"],
                    "sourceQuery": query_text,
                },
            )

            # Connect to related past queries using pgvector
            related = await _STATE_MANAGER.find_related(query_text, threshold=0.4, limit=20)
            print(f"[mock] Found {len(related)} related chunks")

            if related:
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
                    if not chunk_sid or chunk_sid == session["id"]:
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
                                source_node_id=db_node["id"],
                                target_node_id=related_node["id"],
                                relationship="related",
                                session_id=session["id"],
                                weight=round(sim, 3),
                            )
                            print(f"[mock] created edge weight={sim:.3f}")
                        except Exception as e:
                            print(f"[mock] edge failed: {e}")
        except Exception:
            pass
    except Exception:
        pass

    if session is None:
        return {"error": "Failed to create session"}

    return ResearchResponse.model_validate({
        "sessionId": session.get("id"),
        "sessionTitle": query_text,
        "synthesis": synthesis_data,
        "connectedIdeas": ideas_data,
        "applications": apps_data,
        "hypotheses": hypotheses_data,
        "graphData": graph_data,
    })
