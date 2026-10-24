
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from database import engine
from models import InvestigationCase
from schemas import CaseCreate, CaseResponse, CaseUpdate


# Database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


# FastAPI application
app = FastAPI(
    title="Fraud Investigation API",
    description="API for managing fraud investigation cases.",
    version="1.0.0",
)


# Allow the local React/Vite frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Database dependency
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DatabaseSession = Annotated[Session, Depends(get_db)]


# Health check
@app.get("/")
def root():
    return {
        "message": "Fraud Investigation API is running",
        "docs": "/docs",
    }


# Create an investigation case
@app.post(
    "/cases",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_case(
    case_data: CaseCreate,
    db: DatabaseSession,
):
    case = InvestigationCase(**case_data.model_dump())

    db.add(case)
    db.commit()
    db.refresh(case)

    return case


# List investigation cases, optionally filtered by status
@app.get("/cases", response_model=list[CaseResponse])
def list_cases(
    db: DatabaseSession,
    case_status: Annotated[
        str | None,
        Query(alias="status"),
    ] = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    query = select(InvestigationCase)

    if case_status is not None:
        allowed_statuses = {
            "NEW",
            "UNDER_INVESTIGATION",
            "CLOSED",
        }

        if case_status not in allowed_statuses:
            raise HTTPException(
                status_code=400,
                detail="Invalid case status.",
            )

        query = query.where(
            InvestigationCase.status == case_status
        )

    query = (
        query
        .order_by(InvestigationCase.created_at.desc())
        .offset(skip)
        .limit(limit)
    )

    return db.scalars(query).all()


# Retrieve one investigation case
@app.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: int,
    db: DatabaseSession,
):
    case = db.get(InvestigationCase, case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    return case


# Update an investigation case
@app.patch("/cases/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: int,
    case_data: CaseUpdate,
    db: DatabaseSession,
):
    case = db.get(InvestigationCase, case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    changes = case_data.model_dump(exclude_unset=True)

    # Closed cases cannot be modified
    if case.status == "CLOSED" and changes:
        raise HTTPException(
            status_code=400,
            detail="Closed cases cannot be modified.",
        )

    new_status = changes.get("status", case.status)
    new_decision = changes.get("decision", case.decision)

    # Require a final decision before closing a case
    if new_status == "CLOSED" and new_decision is None:
        raise HTTPException(
            status_code=400,
            detail="A final decision is required to close a case.",
        )

    # Recording a decision moves an open case into investigation
    if new_decision is not None and new_status != "CLOSED":
        changes["status"] = "UNDER_INVESTIGATION"

    for field, value in changes.items():
        setattr(case, field, value)

    db.commit()
    db.refresh(case)

    return case
