
# PURPOSE:
#   Handles signal retrieval endpoints:
#   GET /api/v1/signals          → list all signals
#   GET /api/v1/signals/{id}     → get one specific signal
import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from api.schemas import SignalResponse
from api.dependencies import get_hitl_gate
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)
router = APIRouter()


async def _get_pool():
    """Creates a direct asyncpg pool for signal queries."""
    return await asyncpg.create_pool(
        host=settings.db_host, port=settings.db_port,
        database=settings.db_name, user=settings.db_user,
        password=settings.db_password,
        ssl="require" if settings.db_ssl else None,
        min_size=1, max_size=5,
    )


@router.get(
    "/signals",
    summary="List all signals",
    description="Returns all generated signals with optional filters.",
    tags=["Signals"],
)
async def list_signals(
    agent: str = Query(default=None, description="Filter by agent name"),
    signal_type: str = Query(default=None, description="Filter by signal type"),
    status: str = Query(default=None, description="Filter by status: approved, pending, rejected"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum signals to return"),
):
    """
    Returns all signals from the signals table.

    Supports filtering by agent, signal_type, and status.
    Results ordered newest first.
    """

    # A local UI can be useful before Postgres has been provisioned.  Return
    # an empty collection in that state so the dashboard shows its normal
    # empty view instead of surfacing a database connection failure as a 500.
    if not settings.database_is_configured:
        return []

    pool = await _get_pool()

    try:
        async with pool.acquire() as conn:

            conditions = []
            params     = []
            idx        = 1

            if agent:
                conditions.append(f"agent = ${idx}")
                params.append(agent)
                idx += 1

            if signal_type:
                conditions.append(f"signal_type = ${idx}")
                params.append(signal_type)
                idx += 1

            if status:
                conditions.append(f"status = ${idx}")
                params.append(status)
                idx += 1

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            params.append(limit)
            query = f"""
                SELECT signal_id, nct_id, agent, signal_type,
                       summary, confidence, status,
                       created_at::text as created_at
                FROM signals
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${idx}
            """

            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    finally:
        await pool.close()


@router.get(
    "/signals/{signal_id}",
    summary="Get one signal by ID",
    tags=["Signals"],
)
async def get_signal(signal_id: str):
    """Returns the complete details of one specific signal."""

    if not settings.database_is_configured:
        raise HTTPException(
            status_code=404,
            detail="No signals are available because the database is not configured.",
        )

    pool = await _get_pool()

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT signal_id, nct_id, agent, signal_type,
                       summary, evidence, confidence, status,
                       created_at::text as created_at
                FROM signals
                WHERE signal_id = $1
                """,
                signal_id,
            )

        if not row:
            raise HTTPException(
                status_code=404,
                detail=f"Signal {signal_id} not found.",
            )

        return dict(row)

    finally:
        await pool.close()
