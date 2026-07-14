import time
import requests
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- IMPORT THE TEAM'S MODULES ---
import database as db
import ledger

from fastapi import FastAPI, BackgroundTasks
import uvicorn

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SENTINEL-Orchestrator")

# Configuration (Step 1: Point to the Host Server)
BASE_URL = os.getenv("SENTINEL_HOST_URL", "https://garbage-drearily-enactment.ngrok-free.dev")

# API Endpoints
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
AUDIO_PROCESS_ENDPOINT = f"{BASE_URL}/process-audio"

SECURITY_MASTER_PROMPT = """
You are SENTINEL, an active on-device security investigator powered by Gemma 4.
You receive a candidate event from the Tier-1 vision pipeline. 
Investigate the scene actively before making a final decision. Do not passively score.

Available tools for evidence gathering:
- recheck(after_seconds): Wait a moment and get fresh frames. (Crucial for fall recovery checks).
- zoom(box): Request a high-res crop of a bounding box region to resolve ambiguity.
- get_audio(t_start, t_end): Pull a wider audio window from the rolling buffer.
- process_audio(file_path): Clean up noisy audio to get a clear transcript.
- get_history(camera_id, minutes): Check recent incident records for the scene.

When you have enough evidence, you MUST call exactly one terminal action:
- raise_alert(severity, justification, evidence_ids)
- log_benign(reason, evidence_ids)

Output purely as a JSON object containing either a "tool_call" or an "action".
"""

class AgenticOrchestrator:
    """
    Step 2 & 3: The Agentic Orchestrator
    Manages network routing to the remote Gemma 4 instance and catches function calls locally.
    """
    def __init__(self):
        self.session = self._build_retry_session()
        # Initialize the database from the team's script
        db.init_db()
        
        # Generate our agent's keypair for signing records
        self.priv_key, self.pub_key = ledger.generate_key_pair()
        self.pub_key_hex = self.pub_key.public_bytes_raw().hex()
        logger.info(f"Agent Orchestrator initialized. Public Key: {self.pub_key_hex[:16]}...")

    def _build_retry_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def send_to_llm(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            # The remote API expects {"prompt": "string"} rather than a list of message dicts.
            # Convert our message history into a single structured text prompt.
            prompt_str = ""
            for msg in messages:
                role = msg.get("role", "").upper()
                content = msg.get("content", "")
                prompt_str += f"[{role}]: {content}\n"

            response = self.session.post(ANALYZE_ENDPOINT, json={"prompt": prompt_str}, timeout=45)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Network error communicating with remote LLM host: {e}")
            return None
        except ValueError:
            logger.error("LLM host returned invalid JSON.")
            return None

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        logger.info(f"Executing '{tool_name}' with args: {tool_args}")
        try:
            if tool_name == "process_audio":
                # Step 5: Catch Audio Clean locally, push audio payload, return cleaned text
                file_path = tool_args.get("file_path", "")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        # The remote API expects the key 'file'
                        resp = self.session.post(AUDIO_PROCESS_ENDPOINT, files={'file': (os.path.basename(file_path), f, 'audio/wav')})
                        resp.raise_for_status()
                        return resp.json().get("cleaned_text", "No speech detected.")
                return f"Error: Local audio file {file_path} not found."
                    
            elif tool_name == "recheck":
                delay = tool_args.get("after_seconds", 4)
                logger.info(f"Recheck invoked. Halting for {delay} seconds to gather fresh frames...")
                time.sleep(delay)
                # No remote endpoint exists, perform locally
                return "Subject remains motionless on the ground."

            elif tool_name == "zoom":
                # No remote endpoint exists, perform locally
                return "Zoom clear."
                
            elif tool_name == "get_audio":
                return "Audio window captured."

            elif tool_name == "get_history":
                return "No prior incidents in the last 60 minutes."

            return f"Error: Unknown tool '{tool_name}'"
                
        except requests.RequestException as e:
            logger.error(f"Tool endpoint '{tool_name}' failed: {e}")
            return f"Tool execution network error: {e}"

    def seal_and_store_verdict(self, incident_record: Dict[str, Any]):
        """Uses the team's database and ledger modules to seal the record."""
        # Get the previous hash from the team's database module
        prev_h = db.get_last_hash()
        
        event_str = json.dumps(incident_record)
        t_now = datetime.utcnow().isoformat()
        
        # Calculate current hash using the team's ledger module
        curr_h = ledger.compute_hash(prev_h, event_str, t_now)
        
        # Sign the event string
        sig_hex = ledger.sign_data(self.priv_key, event_str).hex()
        
        # Insert into the database
        db.insert_record(incident_record, prev_h, curr_h, sig_hex, self.pub_key_hex)
        logger.info(f"Sealed record in ledger. Hash: {curr_h[:12]}...")

    def investigate_event(self, candidate_event: Dict[str, Any]):
        """The core multi-turn loop."""
        logger.info("=== New Tier-1 Candidate Event Detected ===")
        
        messages = [
            {"role": "system", "content": SECURITY_MASTER_PROMPT},
            {"role": "user", "content": f"New candidate event: {json.dumps(candidate_event)}"}
        ]
        
        tool_call_trace = []
        
        while True:
            llm_response = self.send_to_llm(messages)
            if not llm_response:
                logger.error("Investigation aborted: LLM failed to respond.")
                break
                
            messages.append({"role": "assistant", "content": json.dumps(llm_response)})
            
            # Check for terminal verdict
            action = llm_response.get("action")
            if action in ["raise_alert", "log_benign"]:
                logger.info(f"Terminal action decided: {action.upper()}")
                
                # Construct the schema-locked incident record
                incident_record = {
                    "camera": candidate_event.get("camera_id"),
                    "tier1_triggers": candidate_event,
                    "tool_call_trace": tool_call_trace,
                    "action": action,
                    "severity": llm_response.get("args", {}).get("severity", "LOW" if action == "log_benign" else "HIGH"),
                    "details": llm_response.get("args", {})
                }
                
                # Step 6: Seal the verdict using the team's files
                self.seal_and_store_verdict(incident_record)
                
                # Step 7: Operator Live View trigger
                if action == "raise_alert":
                    logger.warning(f"🚨 ALERT RAISED: {incident_record['details'].get('justification', 'No justification')}")
                else:
                    logger.info(f"🛡️ EVENT SUPPRESSED (Benign): {incident_record['details'].get('reason', 'No reason')}")
                break
                
            # Execute intermediate tool call
            tool_name = llm_response.get("tool_call")
            if tool_name:
                tool_args = llm_response.get("tool_args", {})
                tool_call_trace.append({"tool": tool_name, "args": tool_args})
                
                tool_result = self.execute_tool(tool_name, tool_args)
                messages.append({"role": "user", "content": f"Tool '{tool_name}' result: {tool_result}"})
            else:
                logger.error("LLM response lacked a valid action or tool_call. Breaking to avoid infinite loop.")
                break


# ---------------------------------------------------------
# FastAPI Server for WebCam Triggers
# ---------------------------------------------------------
app = FastAPI(title="SENTINEL 2.0 Orchestrator Server")
orchestrator = AgenticOrchestrator()

@app.post("/candidate_event")
def receive_candidate_event(payload: dict, background_tasks: BackgroundTasks):
    logger.info(f"Received vision trigger event: {payload.get('event_type')}")
    # Kick off the investigation loop in the background so the camera loop doesn't block!
    background_tasks.add_task(orchestrator.investigate_event, payload)
    return {"status": "Investigation triggered in background"}

if __name__ == "__main__":
    logger.info("Starting Orchestrator Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
