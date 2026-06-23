from __future__ import annotations

import asyncio

from backend.core.database import (
    create_session,
    get_session,
    get_sessions,
    save_query,
    get_queries,
    save_research_chunk,
    match_chunks,
    upsert_knowledge_node,
    create_knowledge_edge,
    get_knowledge_graph,
    get_embedding,
)

TEMP_USER_ID = "30435ee7-e228-4d86-9b14-59bf6cdc27be"


class AsyncAgentStateManager:
    def __init__(self, user_id=None):
        self.active_sessions = {}
        self.user_id = user_id or TEMP_USER_ID

    async def new_session(self, title):
        session = await create_session(self.user_id, title)
        self.active_sessions[session["id"]] = session
        return session

    async def list_sessions(self):
        return await get_sessions(self.user_id)

    async def get_session(self, session_id):
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        return await get_session(session_id)

    async def add_query(self, session_id, query_text):
        return await save_query(session_id, query_text)

    async def get_history(self, session_id):
        return await get_queries(session_id)

    async def save_chunk(self, query_id, session_id, chunk_type, content, embed=True, metadata=None):
        embedding = None
        if embed and content:
            try:
                embedding = await asyncio.to_thread(get_embedding, content)
            except Exception:
                print(f"Embedding failed for {chunk_type} chunk, saving without vector")
        return await save_research_chunk(
            query_id=query_id,
            session_id=session_id,
            user_id=self.user_id,
            chunk_type=chunk_type,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )

    async def find_related(self, query_text, threshold=0.7, limit=10):
        try:
            embedding = await asyncio.to_thread(get_embedding, query_text)
            return await match_chunks(embedding, self.user_id, threshold, limit)
        except Exception:
            return []

    async def add_graph_node(self, label, node_type, properties=None):
        return await upsert_knowledge_node(self.user_id, label, node_type, properties)

    async def add_graph_edge(self, source_node_id, target_node_id, relationship, session_id=None, weight=1.0):
        return await create_knowledge_edge(
            self.user_id, source_node_id, target_node_id, relationship, session_id, weight
        )

    async def get_graph(self):
        return await get_knowledge_graph(self.user_id)

    async def connect_related_queries(self, new_node_id, session_id, query_text):
        """Find semantically similar past queries and connect them in the knowledge graph."""
        from backend.core.database import get_supabase, get_embedding

        def connect():
            query_embedding = get_embedding(query_text)
            db = get_supabase()

            result = db.rpc("match_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.65,
                "match_count": 20,
                "p_user_id": self.user_id,
            }).execute()

            related_chunks = result.data or []

            related_session_ids = set()
            for chunk in related_chunks:
                if chunk.get("session_id") and chunk["session_id"] != session_id:
                    related_session_ids.add(chunk["session_id"])

            if not related_session_ids:
                return

            nodes_result = db.table("knowledge_nodes") \
                .select("id, properties") \
                .eq("user_id", self.user_id) \
                .eq("node_type", "concept") \
                .execute()

            session_node_map = {}
            for node in (nodes_result.data or []):
                props = node.get("properties", {}) or {}
                node_session_id = props.get("sessionId") or props.get("session_id")
                if node_session_id in related_session_ids:
                    session_node_map[node_session_id] = node["id"]

            for rel_session_id in related_session_ids:
                target_node_id = session_node_map.get(rel_session_id)
                if not target_node_id:
                    continue

                session_similarities = [
                    c["similarity"] for c in related_chunks
                    if c.get("session_id") == rel_session_id
                ]
                avg_similarity = sum(session_similarities) / len(session_similarities) if session_similarities else 0

                if avg_similarity > 0.65:
                    try:
                        db.table("knowledge_edges").insert({
                            "user_id": self.user_id,
                            "source_node_id": new_node_id,
                            "target_node_id": target_node_id,
                            "relationship": "related",
                            "weight": round(avg_similarity, 3),
                            "session_id": session_id,
                        }).execute()
                    except Exception:
                        pass

        try:
            await asyncio.wait_for(asyncio.to_thread(connect), timeout=5.0)
        except asyncio.TimeoutError:
            print("connect_related_queries timed out after 5s")
        except Exception as e:
            print(f"connect_related_queries error: {e}")

    async def find_discoveries(self, query_text, current_session_id, threshold=0.75, max_results=5):
        from backend.core.database import get_supabase, get_embedding

        try:
            current_embedding = await asyncio.to_thread(get_embedding, query_text)
            db = get_supabase()

            result = db.rpc("match_chunks", {
                "query_embedding": current_embedding,
                "match_threshold": threshold,
                "match_count": max_results * 3,
                "p_user_id": self.user_id,
            }).execute()

            if not result.data:
                return []

            # Deduplicate by session — keep highest similarity per session
            best_per_session = {}
            for match in result.data:
                content = match.get("content", "").strip()
                if content == query_text.strip():
                    continue
                sid = match.get("session_id")
                if not sid or sid == current_session_id:
                    continue
                sim = match.get("similarity", 0)
                if sid not in best_per_session or sim > best_per_session[sid]["similarity"]:
                    best_per_session[sid] = match

            # Score by surprise: high similarity + different chunk type = more surprising
            discoveries = []
            for sid, match in best_per_session.items():
                similarity = match.get("similarity", 0)
                chunk_type = match.get("chunk_type", "synthesis")
                # Cross-type matches are more surprising than same-type
                type_bonus = 1.2 if chunk_type != "synthesis" else 1.0
                surprise = similarity * type_bonus

                discoveries.append({
                    "content": match["content"][:200],
                    "similarity": round(similarity, 3),
                    "surprise_score": round(surprise, 3),
                    "session_id": sid,
                    "chunk_type": chunk_type,
                    "connection_type": "cross_disciplinary" if similarity > 0.8 else "related",
                })

            discoveries.sort(key=lambda x: x["surprise_score"], reverse=True)
            return discoveries[:max_results]

        except Exception as e:
            print(f"find_discoveries error: {e}")
            return []
