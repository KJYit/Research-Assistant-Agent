"""
api.py — FastAPI backend for the Research Assistant Agent.

Exposes a single POST endpoint:
    POST /api/research   { topic, thread_id } -> { answer, sources, thread_id }

CORS is enabled so the React frontend (http://localhost:5173) can freely
communicate with this server (http://localhost:8000).

Run with:
    uvicorn api:app --reload
"""

import sys

# Force UTF-8 output so Unicode characters in print() calls don't crash on
# Windows terminals that default to cp1252 encoding.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import invoke_research_turn
from tools.rate_limit import enforce_research_limit
from tools.research_budget import ResearchDeadlineExceeded

logger = logging.getLogger("uvicorn.error.research")
_RESEARCH_TIMEOUT_SECONDS = 75

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Research Assistant Agent API",
    description="A LangGraph-powered multi-turn research agent backed by Gemini and Tavily.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://research-assistant-agent-ten.vercel.app",
        *[origin.strip().rstrip("/") for origin in os.getenv("RESEARCH_ALLOWED_ORIGINS", "").split(",")
          if origin.strip().rstrip("/")],
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str
    thread_id: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[str]
    thread_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Research Agent API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/research", response_model=ResearchResponse)
def research(req: ResearchRequest, request: Request):
    """
    Execute one research turn for the given topic and thread.

    The thread_id enables LangGraph's MemorySaver to restore conversation
    state between turns — pass the same thread_id for follow-up questions.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    enforce_research_limit(request)
    try:
        final_state = invoke_research_turn(req.topic, req.thread_id, timeout_seconds=_RESEARCH_TIMEOUT_SECONDS)
    except ResearchDeadlineExceeded as exc:
        logger.info("research_turn_failed reason=deadline_exceeded")
        raise HTTPException(504, "The research request timed out. Please retry with a narrower question.") from exc
    except Exception as exc:
        logger.error("research_turn_failed reason=%s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        ) from exc

    answer  = final_state.get("final_answer", "No answer generated.")
    sources = final_state.get("sources", [])

    return ResearchResponse(
        answer=answer,
        sources=sources,
        thread_id=req.thread_id,
    )
