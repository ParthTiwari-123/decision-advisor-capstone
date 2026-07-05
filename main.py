"""
Decision Advisor — FastAPI Backend
GET  /       → serves index.html
POST /decide → runs the orchestrator and returns the merged report
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load GEMINI_API_KEY from .env before anything else

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agents.orchestrator import run_orchestrator

app = FastAPI(title="Decision Advisor", version="1.0.0")


# ── Request schema ──────────────────────────────────────────────────────────
class DecisionRequest(BaseModel):
    decision: str
    risk_tolerance: str
    priority: str
    timeline: str
    past_decisions: str


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the single-page frontend."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/decide", response_class=JSONResponse)
async def decide(req: DecisionRequest):
    """Run the orchestrator pipeline and return the merged report."""
    try:
        result = await run_orchestrator(
            decision=req.decision,
            risk_tolerance=req.risk_tolerance,
            priority=req.priority,
            timeline=req.timeline,
            past_decisions=req.past_decisions,
        )
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
