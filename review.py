
# PURPOSE:
#   Handles the human review queue endpoints:
#   GET   /api/v1/review/queue     → get all pending review items
#   PATCH /api/v1/review/{id}      → submit a human decision
from fastapi import APIRouter, Depends, HTTPException
from api.schemas import ReviewQueueResponse, ReviewDecisionRequest, ReviewDecisionResponse
from api.dependencies import get_hitl_gate
from graph.hitl import HITLGate
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)
router = APIRouter()


@router.get(
    "/review/queue",
    response_model=ReviewQueueResponse,
    summary="Get pending human review queue",
    description="Returns all signals waiting for human analyst review, "
                "ordered by confidence ascending (most uncertain first).",
    tags=["Human Review"],
)
async def get_review_queue(
    hitl_gate: HITLGate = Depends(get_hitl_gate),
):
    """
    Returns all signals currently in the human review queue.

    Items are ordered by confidence ascending — the most uncertain
    signals appear first, since these need the most urgent attention.
    """

    if not settings.database_is_configured:
        return ReviewQueueResponse(
            queue=[],
            total_pending=0,
            total_approved=0,
            total_rejected=0,
        )

    try:
        queue_items = await hitl_gate.get_review_queue()

        import asyncpg
        pool = await asyncpg.create_pool(
            host=settings.db_host, port=settings.db_port,
            database=settings.db_name, user=settings.db_user,
            password=settings.db_password,
            ssl="require" if settings.db_ssl else None,
            min_size=1, max_size=2,
        )

        async with pool.acquire() as conn:
            pending  = await conn.fetchval("SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'pending'")
            approved = await conn.fetchval("SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'approve'")
            rejected = await conn.fetchval("SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'reject'")

        await pool.close()

        return ReviewQueueResponse(
            queue=queue_items, # type: ignore
            total_pending=pending or 0,
            total_approved=approved or 0,
            total_rejected=rejected or 0,
        )

    except Exception as e:
        logger.error(f"get_review_queue failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/review/{queue_id}",
    response_model=ReviewDecisionResponse,
    summary="Submit a human review decision",
    description="Approve, reject, or edit a queued signal. "
                "Rejections trigger the learning loop — the rejection reason "
                "is written to procedural memory permanently.",
    tags=["Human Review"],
)
async def submit_review_decision(
    queue_id: str,
    request: ReviewDecisionRequest,
    hitl_gate: HITLGate = Depends(get_hitl_gate),
):
    """
    Submits a human analyst's decision on a queued signal.

    APPROVE → signal is accepted, marked approved in database
    EDIT    → signal summary is corrected, marked approved
    REJECT  → signal is rejected AND rejection reason is written
               to procedural memory — the agent learns from this

    The REJECT pathway is the most important — it permanently
    changes how the agent reasons in all future sessions.
    """

    if request.decision not in ("approve", "reject", "edit"):
        raise HTTPException(
            status_code=422,
            detail="decision must be 'approve', 'reject', or 'edit'",
        )

    try:
        result = await hitl_gate.process_human_decision(
            queue_id=queue_id,
            decision=request.decision,
            reviewer=request.reviewer,
            rejection_reason=request.rejection_reason,
            edit_summary=request.edit_summary,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", "Review not found"),
            )

        memory_updated = (
            request.decision == "reject"
            and bool(request.rejection_reason)
        )

        logger.info(
            f"Review decision submitted | "
            f"queue_id={queue_id} | "
            f"decision={request.decision} | "
            f"memory_updated={memory_updated}"
        )

        return ReviewDecisionResponse(
            success=True,
            decision=request.decision,
            signal_id=result.get("signal_id", ""),
            queue_id=queue_id,
            memory_updated=memory_updated,
            message=(
                "Procedural memory updated — agent will reason differently "
                "in all future sessions."
                if memory_updated else
                f"Signal {request.decision}d successfully."
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"submit_review_decision failed | error={e}")
        raise HTTPException(status_code=500, detail=str(e))
