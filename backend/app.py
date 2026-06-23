from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.routers.query import router as query_router
from backend.routers.research import router as research_router
from backend.routers.sessions import router as sessions_router
from backend.routers.mock import router as mock_router
from backend.routers.graph_search import router as graph_search_router
from backend.routers.auth import router as auth_router

# Load .env at import time so GEMINI_API_KEY is available everywhere.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Resolve the project-root index.html once.
_INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


class DisconnectionCleanupMiddleware:
    """Cancel active streaming work as soon as the HTTP client disconnects."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_task = asyncio.current_task()
        state: dict[str, Any] = scope.setdefault("state", {})
        state["disconnected"] = False

        def cancel_request_tree() -> None:
            state["disconnected"] = True
            if request_task is not None and not request_task.done():
                request_task.cancel()

        async def receive_with_disconnect_watch() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                cancel_request_tree()
                raise asyncio.CancelledError
            return message

        async def send_with_disconnect_watch(message: Message) -> None:
            try:
                await send(message)
            except OSError:
                cancel_request_tree()
                raise asyncio.CancelledError

        try:
            await self.app(scope, receive_with_disconnect_watch, send_with_disconnect_watch)
        except asyncio.CancelledError:
            state["disconnected"] = True
            raise


class SecurityHeadersMiddleware:
    """Add security headers to every response."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.extend([
                    (b"X-Content-Type-Options", b"nosniff"),
                    (b"X-Frame-Options", b"DENY"),
                    (b"Content-Security-Policy",
                     b"default-src 'self'; "
                     b"script-src 'self' 'unsafe-inline' https://artifactcdn.diabrowser.engineering; "
                     b"style-src 'self' 'unsafe-inline'; "
                     b"img-src 'self' data:; "
                     b"connect-src 'self'; "
                     b"frame-ancestors 'none';"),
                ])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RateLimitMiddleware:
    """Simple in-memory rate limiter for /api/research: 10 requests/minute per IP."""

    def __init__(self, app: ASGIApp):
        self.app = app
        self._history = defaultdict(list)
        self._limit = 10
        self._window = 60

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not path.startswith("/api/research"):
            await self.app(scope, receive, send)
            return

        client = scope.get("client", ("0.0.0.0", 0))[0]
        now = time.time()
        self._history[client] = [t for t in self._history[client] if now - t < self._window]

        if len(self._history[client]) >= self._limit:
            msg = JSONResponse(
                {"detail": "Rate limit exceeded. Max 10 requests per minute."},
                status_code=429,
            )
            raw_headers = [(b"content-type", b"application/json")]
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": raw_headers,
            })
            body = b'{"detail":"Rate limit exceeded. Max 10 requests per minute."}'
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        self._history[client].append(now)
        await self.app(scope, receive, send)


class RequestBodySizeMiddleware:
    """Reject requests with body larger than 10 KB."""

    def __init__(self, app: ASGIApp, max_size: int = 10240):
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            size = int(content_length)
            if size > self.max_size:
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail":"Request body too large. Max 10 KB."}',
                })
                return

        await self.app(scope, receive, send)


app = FastAPI(title="MERIDIAN API", version="0.1.0")

# ── Middleware (order matters: outermost first) ───────────────
app.add_middleware(RequestBodySizeMiddleware, max_size=10240)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(DisconnectionCleanupMiddleware)

# ── Routers ──────────────────────────────────────────────────
app.include_router(query_router)
app.include_router(research_router)
app.include_router(sessions_router)
app.include_router(mock_router)
app.include_router(graph_search_router)
app.include_router(auth_router)


# ── Serve the frontend ───────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the MERIDIAN single-page frontend."""
    return FileResponse(_INDEX_HTML, media_type="text/html")


@app.get("/api/papers/search")
async def search_papers_endpoint(q: str = ""):
    from backend.core.papers import search_papers
    if not q:
        return {"papers": []}
    papers = await search_papers(q, limit=5)
    return {"papers": papers}


@app.get("/api/discoveries")
async def get_discoveries(q: str = "", session_id: str = ""):
    from backend.core.state import AsyncAgentStateManager
    if not q:
        return {"discoveries": []}
    mgr = AsyncAgentStateManager()
    discoveries = await mgr.find_discoveries(q, session_id)
    return {"discoveries": discoveries}
