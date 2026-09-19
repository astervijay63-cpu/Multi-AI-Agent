# PURPOSE:
#   Serves the compiled React frontend (frontend/dist) from the same
#   FastAPI process that serves the API — one server, one origin,
#   no CORS, no second port to deploy.

# HOW IT FITS TOGETHER:
#   `npm run build` in frontend/ emits a hashed bundle into frontend/dist.
#   mount_frontend() is called at the END of main.py, after every router
#   is registered. Starlette matches routes in registration order, so the
#   real API paths always win and only unmatched paths fall through to
#   the SPA.

# IF THE BUNDLE IS NOT BUILT:
#   mount_frontend() logs and returns without mounting anything. The API
#   still runs normally — a missing frontend is never a startup failure.

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.logging_config import setup_logging

logger = setup_logging(__name__)

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
INDEX_HTML    = FRONTEND_DIST / "index.html"

# Paths that belong to the API and must never be answered with index.html.
# Without this, a typo'd endpoint would return 200 and a page of HTML,
# which is far harder to debug than a plain 404.
API_PREFIXES = ("api/", "docs", "redoc", "openapi.json")


def frontend_is_built() -> bool:
    """True when a compiled bundle exists on disk and can be served."""
    return INDEX_HTML.is_file()


def mount_frontend(app: FastAPI) -> None:
    """
    Mounts the built React app onto the given FastAPI application.

    Call this LAST, after all routers are included — the catch-all route
    registered here would otherwise shadow every endpoint declared after it.
    """

    if not frontend_is_built():
        logger.warning(
            f"Frontend bundle not found at {FRONTEND_DIST} — "
            "serving API only. Run 'npm install && npm run build' in frontend/."
        )
        return

    # Vite emits every hashed JS/CSS chunk into dist/assets. Mounting that
    # directory directly lets Starlette stream them without going through
    # the catch-all, and gives correct content types for free.
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """
        Serves a real file when the path names one, and index.html otherwise.

        The fallback is what makes client-side routing work: a deep link
        like /signals is not a file on disk, so the browser needs the app
        shell back and the app reads the URL itself.
        """

        if full_path.startswith(API_PREFIXES):
            return JSONResponse(
                {"detail": f"Not Found: /{full_path}"},
                status_code=404,
            )

        if full_path:
            candidate = (FRONTEND_DIST / full_path).resolve()
            # Resolve first, then confirm the result is still inside dist.
            # A request for ../../.env would otherwise escape the bundle
            # directory and hand out project files.
            if (
                candidate.is_file()
                and candidate.is_relative_to(FRONTEND_DIST.resolve())
            ):
                return FileResponse(candidate)

        return FileResponse(INDEX_HTML)

    logger.info(f"Frontend mounted from {FRONTEND_DIST}")
