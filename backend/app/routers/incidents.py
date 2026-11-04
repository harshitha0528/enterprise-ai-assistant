from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.services.agent_service import investigate_incident
from app.services.log_service import (
    load_entries, parse_log, problem_timeline, problems_between,
    save_log, search_logs, summarize_log,
)

router = APIRouter(prefix="/incidents", tags=["Incident Intelligence"])

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB


class InvestigationRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


def get_entries_or_404(log_id: str) -> list[dict]:
    try:
        return load_entries(log_id)
    except ValueError:
        raise HTTPException(400, "Invalid log id")
    except FileNotFoundError:
        raise HTTPException(404, "Log not found")


@router.get("/")
def get_incidents():
    return {"message": "Incident Intelligence module is ready"}


@router.post("/analyze")
def analyze_log(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith((".log", ".txt")):
        raise HTTPException(400, "Only .log and .txt files are allowed")

    content = file.file.read(MAX_LOG_SIZE + 1)
    if len(content) > MAX_LOG_SIZE:
        raise HTTPException(400, "Log file cannot exceed 5 MB")

    entries = parse_log(content.decode("utf-8", errors="replace"))
    if not entries:
        raise HTTPException(422, "No log lines matched the expected format")

    log_id = save_log(content)
    return {"log_id": log_id, "summary": summarize_log(entries)}


@router.get("/{log_id}/search")
def search(log_id: str, keyword: str):
    return search_logs(get_entries_or_404(log_id), keyword)


@router.get("/{log_id}/timeline")
def timeline(log_id: str):
    return problem_timeline(get_entries_or_404(log_id))


@router.get("/{log_id}/window")
def window(log_id: str, start: str, end: str):
    return problems_between(get_entries_or_404(log_id), start, end)


@router.post("/{log_id}/investigate")
def investigate(log_id: str, request: InvestigationRequest):
    get_entries_or_404(log_id)  # validates the id before spending LLM tokens
    try:
        return investigate_incident(log_id, request.question)
    except Exception as error:
        raise HTTPException(502, f"Investigation failed: {error}")