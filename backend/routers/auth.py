"""MERIDIAN AUTH + ANALYTICS ENDPOINTS"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.database import get_supabase
from datetime import datetime

router = APIRouter(prefix="/api", tags=["Auth & Analytics"])


class AuthRequest(BaseModel):
    email: str
    password: str


class LoginEvent(BaseModel):
    user_id: str = ""
    email: str = ""
    timestamp: str = ""


@router.post("/auth/signup")
async def signup(req: AuthRequest):
    try:
        db = get_supabase()
        result = db.auth.sign_up({
            "email": req.email,
            "password": req.password,
        })
        if result.user:
            return {
                "user_id": result.user.id,
                "email": result.user.email,
                "message": "Account created successfully"
            }
        raise HTTPException(status_code=400, detail="Signup failed")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"SIGNUP ERROR: {error_msg}")  # This will show in Render logs
        if "already registered" in error_msg.lower():
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        raise HTTPException(status_code=400, detail="Signup failed. Please try again.")


@router.post("/auth/login")
async def login(req: AuthRequest):
    try:
        db = get_supabase()
        result = db.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password,
        })
        if result.user:
            return {
                "user_id": result.user.id,
                "email": result.user.email,
                "message": "Login successful"
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"LOGIN ERROR: {error_msg}")  # This will show in Render logs
        if "invalid" in error_msg.lower() or "credentials" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid email or password")
        raise HTTPException(status_code=400, detail="Login failed. Please try again.")


@router.post("/analytics/login")
async def track_login(event: LoginEvent):
    """Track user logins for analytics."""
    try:
        db = get_supabase()
        try:
            db.table("user_analytics").insert({
                "event_type": "login",
                "user_email": event.email,
                "user_id": event.user_id,
                "timestamp": event.timestamp or datetime.utcnow().isoformat(),
            }).execute()
        except Exception:
            pass
        return {"status": "tracked"}
    except Exception:
        return {"status": "ok"}


@router.get("/analytics/stats")
async def get_stats():
    """Get user and usage statistics."""
    try:
        db = get_supabase()

        sessions = db.table("sessions").select("id", count="exact").execute()
        session_count = sessions.count if hasattr(sessions, 'count') else len(sessions.data)

        users = db.table("sessions").select("user_id").execute()
        unique_users = len(set(row["user_id"] for row in users.data)) if users.data else 0

        queries = db.table("queries").select("id", count="exact").execute()
        query_count = queries.count if hasattr(queries, 'count') else len(queries.data)

        nodes = db.table("knowledge_nodes").select("id", count="exact").execute()
        node_count = nodes.count if hasattr(nodes, 'count') else len(nodes.data)

        total_users = unique_users
        try:
            analytics = db.table("user_analytics").select("user_email").execute()
            if analytics.data:
                total_users = max(total_users, len(set(row["user_email"] for row in analytics.data if row.get("user_email"))))
        except Exception:
            pass

        return {
            "total_users": total_users,
            "total_sessions": session_count,
            "total_queries": query_count,
            "total_knowledge_nodes": node_count,
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {
            "total_users": 0,
            "total_sessions": 0,
            "total_queries": 0,
            "total_knowledge_nodes": 0,
        }
