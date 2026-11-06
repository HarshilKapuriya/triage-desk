"""FastAPI main application entrypoint for Ticket Intelligence Platform.

Registers lifespan handlers, global exception handlers for standardized error
envelopes, and mounts API routers and production frontend static serving.
"""

from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import lifespan
from app.deps_browser import (
    BrowserJsonAuthError,
    CsrfValidationError,
    PayloadTooLargeError,
)
from app.routers.auth import router as auth_router
from app.routers.api_keys import router as api_keys_router
from app.routers.tickets import router as tickets_router
from app.routers.browser_api import router as browser_api_router

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

app = FastAPI(
    title="Ticket Intelligence Platform API",
    version="0.1.0",
    description="Multi-tenant customer support ticket intelligence API.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global Error Formatting Envelopes
# ---------------------------------------------------------------------------

@app.exception_handler(BrowserJsonAuthError)
async def browser_json_auth_exception_handler(request: Request, exc: BrowserJsonAuthError) -> JSONResponse:
    """Clear session cookie and return 401 JSON on browser authentication failure."""
    resp = JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"error": {"code": "unauthenticated", "message": exc.message}},
        headers={"Cache-Control": "no-store"},
    )
    resp.delete_cookie(key="triage_session", path="/")
    return resp


@app.exception_handler(CsrfValidationError)
async def csrf_validation_exception_handler(request: Request, exc: CsrfValidationError) -> JSONResponse:
    """Return 403 Forbidden on failed CSRF token validation."""
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"error": {"code": "csrf_forbidden", "message": exc.message}},
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(PayloadTooLargeError)
async def payload_too_large_exception_handler(request: Request, exc: PayloadTooLargeError) -> JSONResponse:
    """Return 413 Content Too Large when request body exceeds maximum permitted size."""
    return JSONResponse(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        content={"error": {"code": "payload_too_large", "message": exc.message}},
        headers={"Cache-Control": "no-store"},
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure HTTP exceptions adhere to the standard SPEC error envelope."""
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        body = {"error": {"code": exc.detail["code"], "message": exc.detail["message"]}}
    else:
        body = {"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail)}}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic request validation errors into the SPEC error envelope."""
    errors = exc.errors()
    if errors:
        first = errors[0]
        locs = list(first.get("loc", []))
        if locs and locs[0] in ("body", "query", "path", "header"):
            locs = locs[1:]
        field_path = " -> ".join(str(loc) for loc in locs)
        msg = first.get("msg", "Invalid input.")
        message = f"Validation error for '{field_path}': {msg}" if field_path else msg
    else:
        message = "Invalid input payload."

    body = {
        "error": {
            "code": "invalid_input",
            "message": message,
        }
    }
    return JSONResponse(status_code=422, content=body)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitize unexpected server exceptions into standard envelope without raw errors."""
    body = {
        "error": {
            "code": "internal_error",
            "message": "An unexpected internal server error occurred.",
        }
    }
    return JSONResponse(status_code=500, content=body)


# ---------------------------------------------------------------------------
# Mount Routers
# ---------------------------------------------------------------------------

app.include_router(browser_api_router)
app.include_router(auth_router)
app.include_router(api_keys_router)
app.include_router(tickets_router)


# Liveness check
@app.get("/healthz", tags=["Health"], summary="Liveness check")
def healthz() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Production Frontend SPA Static Serving
# ---------------------------------------------------------------------------

if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="spa_assets")


@app.get("/{ui_path:path}", include_in_schema=False)
def serve_spa(ui_path: str) -> Response:
    """Serve built React single-page application for UI routes only.

    Explicitly excludes /browser-api/*, /v1/*, and /assets/* so missing API
    endpoints or static assets always return 404 rather than index.html.
    """
    clean_path = ui_path.strip("/")
    if (
        clean_path.startswith("v1")
        or clean_path.startswith("browser-api")
        or clean_path.startswith("assets")
        or clean_path == "healthz"
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend assets not built.")
