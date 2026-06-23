from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.core.orchestrator import AsynchronousOrchestrator
from backend.core.state import AsyncAgentStateManager
from backend.schemas.agent import QueryRequest


router = APIRouter(prefix="/api/v1/query", tags=["Agent Execution"])

_STATE_MANAGER = AsyncAgentStateManager()


def get_state_manager() -> AsyncAgentStateManager:
    """Return the process-wide agent state manager singleton."""
    return _STATE_MANAGER


@router.post("/stream")
async def stream_query(
    request: QueryRequest,
    state_manager: AsyncAgentStateManager = Depends(get_state_manager),
) -> StreamingResponse:
    orchestrator = AsynchronousOrchestrator(
        session_id=request.session_id,
        state_manager=state_manager,
    )
    return StreamingResponse(
        orchestrator.execute_agent_loop(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
