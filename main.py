
# PURPOSE:
#   The FastAPI application entry point.
#   This is the file that starts the entire API server.

# WHAT THIS FILE DOES:
#   1. Creates the FastAPI app with metadata
#   2. Defines the lifespan — what runs at startup and shutdown
#   3. Mounts all 4 routers at their URL prefixes
#   4. Adds CORS middleware for browser access
#   5. Defines the health check endpoint
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routers import analysis, signals, review, memory, study
from api.dependencies import get_hitl_gate, get_procedural_store
from api.static import mount_frontend, frontend_is_built, INDEX_HTML
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    STARTUP (before yield):
    - Initialises default agent procedures in Postgres
    - Logs that the API is ready

    SHUTDOWN (after yield):
    - Closes all database connection pools cleanly
    """

    logger.info("=" * 60)
    logger.info("MOSAIC API starting up...")
    logger.info("=" * 60)

    if settings.database_is_configured:
        try:
            procedural_store = get_procedural_store()
            await procedural_store.initialise_defaults()
            logger.info("Default agent procedures initialised")

        except Exception as e:
            logger.error(f"Failed to initialise procedures | error={e}")
    else:
        logger.warning("Database is not configured; starting in local UI mode")

    logger.info("MOSAIC API ready | version=0.1.0")
    logger.info("=" * 60)

    yield

    logger.info("MOSAIC API shutting down...")

    try:
        hitl = get_hitl_gate()
        await hitl.close()
        logger.info("HITLGate pool closed")
    except Exception as e:
        logger.error(f"Error closing HITLGate | error={e}")

    logger.info("MOSAIC API shutdown complete")


app = FastAPI(
    title="MOSAIC — Clinical Trial Intelligence API",
    description=(
        "Multi-Agent Operating System for AI Cognition. "
        "Detects research integrity signals across clinical trials "
        "using 6 specialist AI agents running in parallel on GCP."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.api_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    analysis.router,
    prefix="/api/v1",
)

app.include_router(
    signals.router,
    prefix="/api/v1",
)

app.include_router(
    review.router,
    prefix="/api/v1",
)

app.include_router(
    memory.router,
    prefix="/api/v1",
)

app.include_router(
    study.router,
    prefix="/api/v1",
)


@app.get(
    "/api/v1/health",
    summary="System health check",
    description="Returns system status including database connectivity "
                "and queue depth. Called by Cloud Run to verify the "
                "container is healthy before routing traffic.",
    tags=["System"],
)
async def health_check():
    """
    Returns the health status of the MOSAIC system.

    Cloud Run calls this endpoint automatically to check if the
    container is healthy. If this returns non-200, Cloud Run marks
    the instance as unhealthy and stops routing traffic to it.
    """

    import asyncpg
    db_status   = "connected"
    signals_count = 0
    pending_count = 0
    episodes_count = 0

    if not settings.database_is_configured:
        db_status = "not configured"
    else:
        try:
            pool = await asyncpg.create_pool(
                host=settings.db_host, port=settings.db_port,
                database=settings.db_name, user=settings.db_user,
                password=settings.db_password,
                ssl="require" if settings.db_ssl else None,
                min_size=1, max_size=2,
            )
            async with pool.acquire() as conn:
                signals_count  = await conn.fetchval("SELECT COUNT(*) FROM signals") or 0
                pending_count  = await conn.fetchval(
                    "SELECT COUNT(*) FROM hitl_reviews WHERE decision = 'pending'"
                ) or 0
                episodes_count = await conn.fetchval("SELECT COUNT(*) FROM episodes") or 0
            await pool.close()

        except Exception as e:
            db_status = f"disconnected: {str(e)}"
            logger.error(f"Health check DB error | error={e}")

    return {
        "status":   "healthy" if db_status == "connected" else "degraded",
        "app":      "MOSAIC",
        "version":  "0.1.0",
        "database": db_status,
        "details": {
            "signals_in_db":   signals_count,
            "pending_reviews": pending_count,
            "episodes_count":  episodes_count,
            "queue_depth":     pending_count,
        },
    }


@app.get("/api/v1/info", tags=["System"])
async def info():
    """Machine-readable descriptor of this API."""
    return {
        "message": "MOSAIC Clinical Trial Intelligence API",
        "version": "0.1.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }


@app.get("/", include_in_schema=False)
async def root():
    """
    Serves the React console when it has been built, and falls back to
    the JSON descriptor when it has not — so a fresh clone with no
    `npm run build` still gets a useful answer here instead of a 404.
    """

    if frontend_is_built():
        return FileResponse(INDEX_HTML)

    return await info()


# MUST BE LAST.
# mount_frontend registers a catch-all GET /{full_path:path} route, and
# Starlette matches routes in registration order — anything declared
# after this line would be unreachable.
mount_frontend(app)
