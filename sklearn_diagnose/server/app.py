"""
FastAPI application for the sklearn-diagnose chatbot server.

This module provides the web server that powers the interactive chatbot UI.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sklearn_diagnose.core.schemas import DiagnosisReport
from sklearn_diagnose.server.chat_agent import ChatAgent


# Get the static files directory
STATIC_DIR = Path(__file__).parent / "static"


# Global state for the server
_chat_agent: Optional[ChatAgent] = None
_diagnosis_report: Optional[DiagnosisReport] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    history: list


class ReportResponse(BaseModel):
    """Response model for report endpoint."""
    report: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="sklearn-diagnose Chatbot",
    description="Interactive chatbot for discussing ML model diagnosis results",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def set_diagnosis_report(report: DiagnosisReport) -> None:
    """
    Set the diagnosis report for the chatbot.

    This function is called by the launcher to initialize the server state.

    Args:
        report: The DiagnosisReport to discuss
    """
    global _diagnosis_report, _chat_agent
    _diagnosis_report = report
    _chat_agent = ChatAgent(report)


def get_chat_agent() -> ChatAgent:
    """
    Get the current chat agent instance.

    Returns:
        The active ChatAgent

    Raises:
        HTTPException: If the chat agent is not initialized
    """
    if _chat_agent is None:
        raise HTTPException(
            status_code=500,
            detail="Chat agent not initialized. Please start the server with a diagnosis report.",
        )
    return _chat_agent


@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Dict with status and agent initialization state
    """
    return {
        "status": "healthy",
        "agent_initialized": _chat_agent is not None,
    }


@app.get("/api/report")
async def get_report() -> ReportResponse:
    """
    Get the current diagnosis report.

    Returns:
        The diagnosis report as a dictionary

    Raises:
        HTTPException: If no report is available
    """
    if _diagnosis_report is None:
        raise HTTPException(
            status_code=404,
            detail="No diagnosis report available.",
        )

    return ReportResponse(report=_diagnosis_report.to_dict())


@app.get("/api/welcome")
async def get_welcome_message():
    """
    Get the welcome message with diagnosis summary.

    Returns:
        Dict with the welcome message

    Raises:
        HTTPException: If chat agent is not initialized
    """
    agent = get_chat_agent()
    return {"message": agent.get_welcome_message()}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message and return the response.

    Args:
        request: ChatRequest with the user's message

    Returns:
        ChatResponse with the assistant's response and conversation history

    Raises:
        HTTPException: If chat agent is not initialized or message is empty
    """
    agent = get_chat_agent()

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # Generate response
    response = agent.chat(request.message)

    # Return response with updated history
    return ChatResponse(
        response=response,
        history=agent.get_history(),
    )


@app.post("/api/clear")
async def clear_chat():
    """
    Clear the conversation history.

    Returns:
        Dict with success status

    Raises:
        HTTPException: If chat agent is not initialized
    """
    agent = get_chat_agent()
    agent.clear_history()

    return {
        "status": "success",
        "message": "Conversation history cleared.",
    }


@app.get("/api/history")
async def get_history():
    """
    Get the conversation history.

    Returns:
        Dict with the conversation history

    Raises:
        HTTPException: If chat agent is not initialized
    """
    agent = get_chat_agent()

    return {
        "history": agent.get_history(),
    }


# Mount static files (CSS, JS, etc.) from the built frontend
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serve the frontend React app for all non-API routes.

    This catch-all route ensures the React app's routing works correctly.
    For any path that's not an API endpoint, serve the index.html.
    """
    # If it's an API route, let it pass through (should be caught by API routes above)
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    # Serve index.html for all other routes (React handles routing)
    index_path = STATIC_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Frontend not built. Run 'npm run build' in the frontend directory."
        )

    return FileResponse(index_path)
