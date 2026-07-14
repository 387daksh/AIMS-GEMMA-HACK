import time
import requests
import json
import os
import logging
import sqlite3
import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Attempt to import pynacl for Ed25519 signatures (as per SENTINEL 2.0 spec)
try:
    from nacl.signing import SigningKey
    from nacl.encoding import HexEncoder
    HAS_NACL = True
except ImportError:
    HAS_NACL = False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SENTINEL-Orchestrator")

# Configuration (Step 1: Point to the Host Server)
BASE_URL = os.getenv("SENTINEL_HOST_URL", "http://127.0.0.1:5000")

# API Endpoints
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
AUDIO_PROCESS_ENDPOINT = f"{BASE_URL}/process-audio"
RECHECK_ENDPOINT = f"{BASE_URL}/recheck"
ZOOM_ENDPOINT = f"{BASE_URL}/zoom"
GET_AUDIO_ENDPOINT = f"{BASE_URL}/get-audio"
HISTORY_ENDPOINT = f"{BASE_URL}/get-history"

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

class SecureLedger:
    """
    Step 6: Verdict & Lock
    Manages the cryptographic chaining (SHA-256) and signing (Ed25519) of incident records in SQLite.
    """
    def __init__(self, db_path: str = "sentinel_ledger.db"):
        self.db_path = db_path
        self._init_db()
        self._init_keys()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ledger (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    record_json TEXT,
                    prev_hash TEXT,
                    hash TEXT,
                    signature TEXT
                )
            """)

    def _init_keys(self):
        if HAS_NACL:
            # For demo: generate an ephemeral key if none exists.
            # In production: load from TPM / secure element.
            self.signing_key = SigningKey.generate()
            self.verify_key = self.signing_key.verify_key
            logger.info("Ed25519 Cryptographic signing enabled (PyNaCl).")
        else:
            logger.warning("PyNaCl not installed. Falling back to mock signatures. 'pip install pynacl' for real Ed25519.")

    def _get_last_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT hash FROM ledger ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else "0000000000000000000000000000000000000000000000000000000000000000"

    def commit_record(self, incident_record: Dict[str, Any]) -> str:
        record_id = incident_record.get("id", str(uuid.uuid4()))
        timestamp = datetime.utcnow().isoformat()
        
        # Lock schema and serialize
        record_str = json.dumps(incident_record, sort_keys=True)
        prev_hash = self._get_last_hash()
        
        # SHA-256 Chain
        hasher = hashlib.sha256()
        hasher.update(prev_hash.encode('utf-8'))
        hasher.update(record_str.encode('utf-8'))
        current_hash = hasher.hexdigest()
        
        # Ed25519 Signature
        signature_hex = "mock_signature_no_pynacl_installed"
        if HAS_NACL:
            signed = self.signing_key.sign(current_hash.encode('utf-8'), encoder=HexEncoder)
            signature_hex = signed.signature.decode('utf-8')
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO ledger (id, timestamp, record_json, prev_hash, hash, signature) VALUES (?, ?, ?, ?, ?, ?)",
                (record_id, timestamp, record_str, prev_hash, current_hash, signature_hex)
            )
            
        logger.info(f"Ledger Commit: Record {record_id[:8]}... cryptographically sealed. Hash: {current_hash[:8]}...")
        return current_hash


class AgenticOrchestrator:
    """
    Step 2 & 3: The Agentic Orchestrator
    Manages network routing to the remote Gemma 4 instance and catches function calls locally.
    """
    def __init__(self):
        self.session = self._build_retry_session()
        self.ledger = SecureLedger()

    def _build_retry_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def send_to_llm(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            response = self.session.post(ANALYZE_ENDPOINT, json={"messages": messages}, timeout=45)
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
                        resp = self.session.post(AUDIO_PROCESS_ENDPOINT, files={'audio_file': f})
                        resp.raise_for_status()
                        return resp.json().get("cleaned_text", "No speech detected.")
                return f"Error: Local audio file {file_path} not found."
                    
            elif tool_name == "recheck":
                delay = tool_args.get("after_seconds", 4)
                logger.info(f"Recheck invoked. Halting for {delay} seconds to gather fresh frames...")
                time.sleep(delay)
                resp = self.session.post(RECHECK_ENDPOINT, json=tool_args)
                resp.raise_for_status()
                return resp.json().get("result")

            elif tool_name == "zoom":
                resp = self.session.post(ZOOM_ENDPOINT, json=tool_args)
                resp.raise_for_status()
                return resp.json().get("result")
                
            elif tool_name == "get_audio":
                resp = self.session.post(GET_AUDIO_ENDPOINT, json=tool_args)
                resp.raise_for_status()
                return resp.json().get("result")

            elif tool_name == "get_history":
                resp = self.session.post(HISTORY_ENDPOINT, json=tool_args)
                resp.raise_for_status()
                return resp.json().get("result")

            return f"Error: Unknown tool '{tool_name}'"
                
        except requests.RequestException as e:
            logger.error(f"Tool endpoint '{tool_name}' failed: {e}")
            return f"Tool execution network error: {e}"

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
                    "id": str(uuid.uuid4()),
                    "t": datetime.utcnow().isoformat(),
                    "camera": candidate_event.get("camera_id"),
                    "tier1_triggers": candidate_event,
                    "tool_call_trace": tool_call_trace,
                    "action": action,
                    "details": llm_response.get("args", {})
                }
                
                # Step 6: Seal the verdict
                self.ledger.commit_record(incident_record)
                
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
# Mock Vision Pipeline Trigger
# ---------------------------------------------------------
if __name__ == "__main__":
    orchestrator = AgenticOrchestrator()
    
    mock_candidate_event = {
        "event_id": "EVT-883A",
        "timestamp": datetime.utcnow().isoformat(),
        "camera_id": "CAM-EAST-02",
        "motion_score": 0.92,
        "persons_detected": 1,
        "boxes": [[210, 450, 310, 580]],
        "context_hint": "Person on ground, possible fall.",
        "local_audio_buffer": "/tmp/buffer/cam-east-02-last5s.wav"
    }
    
    orchestrator.investigate_event(mock_candidate_event)
