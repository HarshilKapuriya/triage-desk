"""Authentication routers: POST /v1/auth/signup and POST /v1/auth/login.

Implements SPEC.md contracts:
- Single-transaction signup (tenant, admin user, audit log).
- Standardized error envelope: {"error": {"code": "<machine_code>", "message": "<text>"}}.
- Consistent 401 response for unknown email and wrong password.
"""

from typing import Annotated
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import re

from app.db import get_connection
from app.repositories.auth import (
    create_tenant_and_admin_user,
    get_user_by_email,
    EmailAlreadyExistsError,
)
from app.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/v1/auth", tags=["Auth"])

# Basic regex for email format validation without additional dependencies
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class SignupRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200, description="Company/Tenant name.")
    email: str = Field(..., description="Admin email address.")
    password: str = Field(..., min_length=8, description="Password (at least 8 characters).")

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError("Invalid email format.")
        return cleaned

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Company name must not be blank.")
        return cleaned


class SignupResponse(BaseModel):
    tenant_id: str
    user_id: str
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: str = Field(..., description="Registered email address.")
    password: str = Field(..., description="Password.")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new tenant and admin user",
)
def signup(payload: SignupRequest) -> SignupResponse:
    """Register a new tenant company and its primary admin user in one transaction."""
    # Hash password with Argon2
    password_hash = hash_password(payload.password)

    try:
        with get_connection() as conn:
            tenant_id, user_id = create_tenant_and_admin_user(
                conn=conn,
                company_name=payload.company_name,
                normalized_email=payload.email,
                password_hash=password_hash,
            )
    except EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "email_already_registered",
                "message": "Email is already registered.",
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "database_error",
                "message": "Failed to complete signup.",
            },
        ) from exc

    # Generate initial JWT token for admin
    token = create_access_token(user_id=user_id, tenant_id=tenant_id, role="admin")

    return SignupResponse(
        tenant_id=str(tenant_id),
        user_id=str(user_id),
        access_token=token,
        token_type="bearer",
    )


@router.post(
    "/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and issue access token",
)
def login(payload: LoginRequest) -> LoginResponse:
    """Log in with email and password to receive a JWT access token.

    Returns identical 401 responses for unknown email and wrong password to prevent user enumeration.
    """
    invalid_creds_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "invalid_credentials",
            "message": "Invalid email or password.",
        },
    )

    with get_connection() as conn:
        user = get_user_by_email(conn, payload.email)

    if not user:
        raise invalid_creds_exc

    if not verify_password(user.password_hash, payload.password):
        raise invalid_creds_exc

    token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        role=user.role,
    )

    return LoginResponse(
        access_token=token,
        token_type="bearer",
    )

