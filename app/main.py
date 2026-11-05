"""FastAPI main application entrypoint for Ticket Intelligence Platform.

Registers lifespan handlers, global exception handlers for standardized error
envelopes, and mounts routers.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.db import lifespan
from app.routers.auth import router as auth_router
from app.routers.api_keys import router as api_keys_router
from app.routers.tickets import router as tickets_router

app = FastAPI(
    title="Ticket Intelligence Platform API",
    version="0.1.0",
    description="Multi-tenant customer support ticket intelligence API.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global Error Formatting Envelope
# Format: {"error": {"code": "<machine_code>", "message": "<text>"}}
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure HTTP exceptions adhere to the standard SPEC error envelope."""
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        body = {"error": {"code": exc.detail["code"], "message": exc.detail["message"]}}
    else:
        # Fallback for generic string details
        body = {"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail)}}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic request validation errors into the SPEC error envelope.

    Never echoes sensitive field values or raw stack traces.
    """
    # Build clean, user-friendly validation error message without echoing inputs
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


# Mount routers
app.include_router(auth_router)
app.include_router(api_keys_router)
app.include_router(tickets_router)


# Liveness check
@app.get("/healthz", tags=["Health"], summary="Liveness check")
def healthz() -> dict[str, str]:
    """Liveness probe endpoint.

    Returns HTTP 200 with {"status": "ok"}. Does not perform database queries
    or expose internal system details.
    """
    return {"status": "ok"}
