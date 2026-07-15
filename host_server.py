import json
import logging

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

import database as db
import ledger
from orchestrator import AgenticOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SENTINEL-HostServer")

app = FastAPI(title="SENTINEL 2.0 Host Server")

db.init_db()
orchestrator = AgenticOrchestrator()


@app.get("/")
def dashboard():
    return FileResponse("templates/index.html")


@app.post("/candidate-event")
def receive_candidate_event(payload: dict, background_tasks: BackgroundTasks):
    camera = payload.get("camera_id")

    # Single-flight: must acquire the lock synchronously, before scheduling the
    # background task — background_tasks only run after this response is sent,
    # so this is the only point where we can honestly tell the caller "busy".
    if not orchestrator.try_begin_episode(payload):
        logger.warning(f"Rejecting candidate event from {camera}: an investigation is already in progress.")
        return JSONResponse(status_code=409, content={
            "status": "busy",
            "detail": "An investigation is already in progress. This candidate event was dropped.",
        })

    logger.info(f"Received Tier-1 candidate event: {payload.get('event_type')} from {camera}")
    # Runs the agentic investigation in the background so vision_trigger's POST returns immediately.
    background_tasks.add_task(orchestrator.run_investigation, payload)
    return {"status": "investigation_triggered"}


@app.get("/api/ledger-logs")
def ledger_logs():
    records = db.get_all_records(include_suppressed=True)
    out = []
    for record in records:
        rec_id, timestamp, event_data_str, prev_hash, current_hash, signature, pub_key, is_suppressed = record
        try:
            event = json.loads(event_data_str)
        except (json.JSONDecodeError, TypeError):
            event = {}
        out.append({
            "id": rec_id,
            "timestamp": timestamp,
            "camera": event.get("camera", "unknown"),
            "severity": event.get("severity", "LOW"),
            "action": event.get("action", ""),
            "justification": event.get("justification", ""),
            "tool_call_trace": event.get("tool_call_trace", []),
            "tier1_triggers": event.get("tier1_triggers", {}),
            "previous_hash": prev_hash,
            "current_hash": current_hash,
            "is_suppressed": bool(is_suppressed),
        })
    out.reverse()  # newest first
    return JSONResponse(out)


@app.get("/api/verify")
def verify_ledger():
    records = db.get_all_records(include_suppressed=True)
    valid, broken_at, message = ledger.verify_chain_detailed(records)
    return {"valid": valid, "broken_at": broken_at, "message": message}


@app.get("/api/status")
def status():
    """Powers the dashboard's 'investigating...' indicator."""
    return orchestrator.get_status()


if __name__ == "__main__":
    logger.info("Starting SENTINEL Host Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
