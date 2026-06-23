from __future__ import annotations
import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

_env_loaded = False
for _p in [Path(__file__).resolve().parent.parent.parent / '.env',
           Path.home() / 'Meridian' / '.env',
           Path.home() / 'meridian' / '.env',
           Path.cwd() / '.env']:
    if _p.is_file():
        load_dotenv(str(_p), override=True)
        _env_loaded = True
        break
if not _env_loaded:
    load_dotenv()

_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError(
                f"SUPABASE_URL or SUPABASE_ANON_KEY not found. "
                f"Searched .env files. CWD={os.getcwd()}, "
                f"__file__={__file__}"
            )
        _supabase = create_client(url, key)
    return _supabase


async def create_session(user_id, title):
    result = get_supabase().table("sessions").insert({
        "user_id": user_id,
        "title": title,
    }).execute()
    return result.data[0]


async def get_sessions(user_id):
    result = get_supabase().table("sessions") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("updated_at", desc=True) \
        .execute()
    return result.data


async def get_session(session_id):
    result = get_supabase().table("sessions") \
        .select("*") \
        .eq("id", session_id) \
        .single() \
        .execute()
    return result.data


async def update_session_timestamp(session_id):
    get_supabase().table("sessions") \
        .update({"updated_at": "now()"}) \
        .eq("id", session_id) \
        .execute()


async def save_query(session_id, query_text):
    result = get_supabase().table("queries").insert({
        "session_id": session_id,
        "query_text": query_text,
    }).execute()
    await update_session_timestamp(session_id)
    return result.data[0]


async def get_queries(session_id):
    result = get_supabase().table("queries") \
        .select("*") \
        .eq("session_id", session_id) \
        .order("created_at") \
        .execute()
    return result.data


async def save_research_chunk(query_id, session_id, user_id, chunk_type, content, embedding=None, metadata=None):
    row = {
        "query_id": query_id,
        "session_id": session_id,
        "user_id": user_id,
        "chunk_type": chunk_type,
        "content": content,
        "metadata": metadata or {},
    }
    if embedding:
        row["embedding"] = embedding
    result = get_supabase().table("research_chunks").insert(row).execute()
    return result.data[0]


async def match_chunks(query_embedding, user_id, match_threshold=0.7, match_count=10):
    result = get_supabase().rpc("match_chunks", {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "p_user_id": user_id,
    }).execute()
    return result.data


async def upsert_knowledge_node(user_id, label, node_type, properties=None):
    result = get_supabase().table("knowledge_nodes").upsert(
        {
            "user_id": user_id,
            "label": label,
            "node_type": node_type,
            "properties": properties or {},
        },
        on_conflict="user_id,label,node_type",
    ).execute()
    return result.data[0]


async def create_knowledge_edge(user_id, source_node_id, target_node_id, relationship, session_id=None, weight=1.0):
    result = get_supabase().table("knowledge_edges").insert({
        "user_id": user_id,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "relationship": relationship,
        "session_id": session_id,
        "weight": weight,
    }).execute()
    return result.data[0]


async def get_knowledge_graph(user_id):
    nodes = get_supabase().table("knowledge_nodes") \
        .select("*") \
        .eq("user_id", user_id) \
        .execute()
    edges = get_supabase().table("knowledge_edges") \
        .select("*") \
        .eq("user_id", user_id) \
        .execute()
    return {"nodes": nodes.data, "links": edges.data}


async def save_document(user_id, session_id, title, content, doc_format="markdown"):
    result = get_supabase().table("documents").insert({
        "user_id": user_id,
        "session_id": session_id,
        "title": title,
        "content": content,
        "format": doc_format,
    }).execute()
    return result.data[0]


def get_embedding(text):
    import google.genai as genai
    from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return result.embeddings[0].values
