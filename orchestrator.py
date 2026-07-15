import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import database as db
import ledger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SENTINEL-Orchestrator")

# --- Local-only endpoints. No ngrok, no remote hosts. ---
OLLAMA_URL = os.getenv("SENTINEL_OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("SENTINEL_OLLAMA_MODEL", "gemma4:e4b")
AUDIO_SERVICE_URL = os.getenv("SENTINEL_AUDIO_SERVICE_URL", "http://127.0.0.1:5000")
DEFAULT_VISION_TOOL_URL = os.getenv("SENTINEL_VISION_TOOL_URL", "http://127.0.0.1:5001")

MAX_TOOL_CALLS = 6
MAX_PARSE_RETRIES = 2
MAX_EPISODE_SECONDS = float(os.getenv("SENTINEL_MAX_EPISODE_SECONDS", "180"))
WATCHDOG_POLL_SECONDS = 5

SECURITY_MASTER_PROMPT = """
You are SENTINEL, an active on-device security investigator powered by Gemma.
You receive a candidate event from the Tier-1 vision pipeline. Investigate the
scene actively before making a final decision. Do not passively score.

You have exactly these tools:
- recheck(after_seconds): Wait, then get a genuinely fresh observation of the
  same camera (not a replay of the original frame). Use this to check for
  recovery after a fall, or to confirm motion is ongoing.
- zoom(box=[x1,y1,x2,y2], frame_ts=optional): Crop a region from the RAW
  full-resolution frame to resolve visual ambiguity. Coordinates are in the
  same pixel space as the boxes on the candidate event / recheck results.
- get_audio(t_start, t_end): Pull a window from the rolling raw audio buffer.
  Returns acoustic features (peak/RMS amplitude, duration, clip_path) — NOT a
  transcript. Use this first before deciding whether isolation is needed.
- isolate_audio(clip_path): Run SepFormer source separation on a clip_path you
  already obtained from get_audio. Only call this if you judge the raw clip
  too noisy/ambiguous to reason about — do not call it reflexively on every
  audio window.
- get_history(camera_id, minutes): Look up recent signed ledger records for
  this camera in the given time window (severity, timestamp, justification).

Before terminating, prefer to gather at least one piece of confirming evidence
(e.g. zoom to verify what you're actually seeing, recheck to confirm motion
persists) unless the candidate event is already unambiguous either way.
Terminating on the very first turn with zero tool calls should be rare.

WELFARE CONCERNS ARE ALERTS TOO — not just threats. A large share of real
emergencies involve no second person, no weapon, and no aggression at all. Do
not treat "no threat actor" as "no emergency." Concrete grounding:
- A person who is prone or on the ground and has NOT recovered by the time of
  a recheck is a MEDIUM or HIGH welfare concern and must be raised with
  raise_alert — even with no weapon, no aggressor, no shouting, and no other
  threat indicator present. Non-recovery after a fall IS the emergency by
  itself. "No immediate distress visible" is not a valid reason to call this
  benign if the person still has not gotten back up.
- Absence of a weapon or an aggressor is NOT evidence of absence of an
  emergency — it is simply irrelevant to a welfare/medical concern. Do not
  require threat-actor evidence before raising a fall, collapse, or
  unresponsive-person finding.
- When you use recheck on a fallen/prone person, your justification must
  directly state whether they recovered or not. "Person moving in a general
  area" is not an adequate description of a recheck result on someone who was
  down — describe what actually changed (or didn't) between the original
  observation and the recheck, and weigh non-recovery as the primary signal,
  not the absence of other danger signs.

When you have enough evidence, you MUST call exactly one terminal action:
- raise_alert(severity, justification, evidence_ids): use this ONLY when your
  own justification describes an actual concern a human should see right now —
  a real hazard, a distress signal, prohibited activity, or a genuine anomaly.
  severity must be MEDIUM, HIGH, or CRITICAL. If you are not confident enough
  to assign at least MEDIUM, that lack of confidence IS your answer — call
  log_benign instead. Never raise_alert "just in case" when your own
  justification says there's no clear threat.
- log_benign(reason, evidence_ids): use this whenever evidence is inconclusive,
  ambiguous, or shows nothing unusual — including when you simply don't have
  enough evidence to be sure. Insufficient evidence is a reason to log_benign,
  not a reason to raise_alert at low severity.

Respond with ONLY a single JSON object, no markdown, no commentary. Either:
{"tool_call": "<name>", "tool_args": {...}}
or:
{"action": "raise_alert" | "log_benign", "args": {...}}
"""


class AgenticOrchestrator:
    """
    The agentic reasoning loop: sends the candidate event + tool results to a
    local Gemma instance via Ollama, executes whichever tool it calls against
    real local services, and terminates every investigation with exactly one
    signed ledger record.
    """

    def __init__(self):
        self.session = self._build_retry_session()
        db.init_db()

        self.priv_key, self.pub_key = ledger.generate_key_pair()
        self.pub_key_hex = self.pub_key.public_bytes_raw().hex()
        logger.info(f"Agent Orchestrator initialized. Public Key: {self.pub_key_hex[:16]}...")

        # Single-flight lock: only one investigation runs at a time. Without this,
        # a burst of candidate events each starts its own concurrent Ollama call —
        # Ollama serializes generation against one loaded model, so concurrent
        # requests queue up inside it and every call gets progressively slower
        # until they all time out.
        self._episode_lock = threading.Lock()

        # Bookkeeping for the watchdog + /api/status, protected by _state_lock
        # (a separate, always-uncontended lock — never held during network I/O).
        # _episode_token increments every time the lock is acquired; it's how a
        # late-finishing worker thread and the watchdog's forced release avoid
        # racing each other (see _end_episode).
        self._state_lock = threading.Lock()
        self._episode_token = 0
        self._episode_started_at: Optional[float] = None
        self._episode_candidate_event: Optional[Dict[str, Any]] = None
        self._episode_active = False

        # Serializes ledger writes so a watchdog-forced record can never
        # interleave with a normal one mid get_last_hash()/insert_record(),
        # which would fork the hash chain.
        self._ledger_lock = threading.Lock()

        threading.Thread(target=self._watchdog_loop, daemon=True).start()
        logger.info(f"Episode watchdog armed (timeout={MAX_EPISODE_SECONDS:.0f}s, poll={WATCHDOG_POLL_SECONDS}s).")

    def try_begin_episode(self, candidate_event: Dict[str, Any]) -> bool:
        """Non-blocking acquire. False means an investigation is already running."""
        if not self._episode_lock.acquire(blocking=False):
            return False
        with self._state_lock:
            self._episode_token += 1
            self._episode_started_at = time.monotonic()
            self._episode_candidate_event = candidate_event
            self._episode_active = True
        return True

    def get_status(self) -> Dict[str, Any]:
        """Powers the dashboard's 'investigating...' indicator."""
        with self._state_lock:
            if not self._episode_active or self._episode_started_at is None:
                return {"busy": False}
            elapsed = time.monotonic() - self._episode_started_at
            camera = (self._episode_candidate_event or {}).get("camera_id")
        return {"busy": True, "camera": camera, "elapsed_seconds": round(elapsed, 1)}

    def _end_episode(self, token: int, reason: str):
        """
        Releases the episode lock exactly once per token. If the watchdog has
        already force-ended this episode (different/cleared token), this is a
        safe no-op — otherwise a truly-stuck worker thread that eventually wakes
        up would release a lock a newer episode is actively using.
        """
        with self._state_lock:
            if self._episode_token != token or not self._episode_active:
                logger.info(f"_end_episode('{reason}') for token {token} skipped — already ended.")
                return
            self._episode_active = False
            self._episode_started_at = None
            self._episode_candidate_event = None
        self._episode_lock.release()
        logger.info(f"Episode lock released ({reason}).")

    def run_investigation(self, candidate_event: Dict[str, Any]):
        """
        Entry point for host_server.py's background task. Caller must have
        already acquired the lock via try_begin_episode(). The finally block
        guarantees the lock is released even if investigate_event raises an
        unexpected exception, so a bug can never permanently wedge the system
        into rejecting every future candidate event.
        """
        with self._state_lock:
            token = self._episode_token
        try:
            self.investigate_event(candidate_event)
        except Exception:
            logger.exception("investigate_event raised an unhandled exception.")
            raise
        finally:
            self._end_episode(token, reason="worker finished")

    def _watchdog_loop(self):
        """
        Safety net for a TRUE hang (network blip, Ollama wedged, etc.) beyond
        what per-request timeouts already catch. Runs for the life of the
        process, polling every WATCHDOG_POLL_SECONDS.
        """
        while True:
            time.sleep(WATCHDOG_POLL_SECONDS)
            with self._state_lock:
                active = self._episode_active
                started_at = self._episode_started_at
                token = self._episode_token
                candidate_event = self._episode_candidate_event
            if not active or started_at is None:
                continue
            elapsed = time.monotonic() - started_at
            if elapsed > MAX_EPISODE_SECONDS:
                logger.error(
                    f"Episode token {token} has been running for {elapsed:.1f}s, exceeding the "
                    f"{MAX_EPISODE_SECONDS:.0f}s safety timeout. Forcing a log_benign fallback and "
                    f"releasing the lock."
                )
                self._force_timeout_fallback(token, candidate_event, elapsed)

    def _force_timeout_fallback(self, token: int, candidate_event: Optional[Dict[str, Any]], elapsed: float):
        try:
            self._finalize(
                candidate_event or {},
                tool_call_trace=[],
                action="log_benign",
                args={
                    "reason": f"Investigation forcibly terminated by the watchdog after running "
                              f"{elapsed:.1f}s, exceeding the {MAX_EPISODE_SECONDS:.0f}s safety timeout. "
                              f"This usually means Ollama or a tool endpoint stopped responding.",
                    "evidence_ids": [],
                },
                turn_timings=[{"kind": "watchdog_timeout", "think": None, "elapsed_seconds": round(elapsed, 2)}],
                episode_start=time.monotonic() - elapsed,
            )
        except Exception:
            logger.exception(f"Watchdog failed to seal a forced-timeout record for token {token}.")
        finally:
            # The original worker thread is NOT killed (Python can't safely do that) — if it
            # eventually does finish on its own, its own _end_episode() call will just no-op
            # against this new token. You may occasionally see two records for one episode
            # after a forced timeout; that's the accepted tradeoff for never staying wedged.
            self._end_episode(token, reason="watchdog timeout")

    def _build_retry_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    @staticmethod
    def _parse_tool_json(content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        return json.loads(text.strip())

    def send_to_llm(self, messages: List[Dict[str, Any]], think: bool) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Returns (parsed_response_or_None, elapsed_seconds). think=False is for
        intermediate tool-selection turns (~4x faster per latency testing);
        think=True is reserved for the turn that actually produces the terminal
        raise_alert/log_benign decision, since that reasoning trace is what gets
        shown in the demo.
        """
        start = time.monotonic()
        logger.info(f"Ollama call started (think={think})")
        try:
            response = self.session.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False, "format": "json", "think": think},
                timeout=180 if think else 60,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = self._parse_tool_json(content)
            elapsed = time.monotonic() - start
            logger.info(f"Ollama call finished (think={think}) in {elapsed:.2f}s")
            return parsed, elapsed
        except requests.RequestException as e:
            elapsed = time.monotonic() - start
            logger.error(f"Ollama call failed (think={think}) after {elapsed:.2f}s: network error: {e}")
            return None, elapsed
        except ValueError as e:
            elapsed = time.monotonic() - start
            logger.error(f"Ollama call failed (think={think}) after {elapsed:.2f}s: JSON parse error: {e}")
            return None, elapsed

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any], candidate_event: Dict[str, Any]) -> Any:
        logger.info(f"Executing '{tool_name}' with args: {tool_args}")
        tool_server = candidate_event.get("tool_server_url", DEFAULT_VISION_TOOL_URL)
        try:
            if tool_name == "recheck":
                after_seconds = tool_args.get("after_seconds", 3)
                resp = self.session.get(
                    f"{tool_server}/recheck",
                    params={"after_seconds": after_seconds},
                    timeout=after_seconds + 10,
                )
                resp.raise_for_status()
                data = resp.json()
                return {
                    "frame_ts": data.get("frame_ts"),
                    "person_count": data.get("person_count"),
                    "boxes": data.get("boxes"),
                    "confidences": data.get("confidences"),
                    "motion_score": data.get("motion_score"),
                }

            elif tool_name == "zoom":
                box = tool_args.get("box") or [
                    tool_args.get("x1"), tool_args.get("y1"), tool_args.get("x2"), tool_args.get("y2")
                ]
                if not box or len(box) != 4 or any(v is None for v in box):
                    return {"error": "zoom requires a box=[x1,y1,x2,y2]"}
                params = {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
                if tool_args.get("frame_ts") is not None:
                    params["frame_ts"] = tool_args["frame_ts"]
                resp = self.session.get(f"{tool_server}/zoom", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return {"frame_ts": data.get("frame_ts"), "box": data.get("box")}

            elif tool_name == "get_audio":
                resp = self.session.get(
                    f"{tool_server}/audio",
                    params={"t_start": tool_args.get("t_start"), "t_end": tool_args.get("t_end")},
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "isolate_audio":
                clip_path = tool_args.get("clip_path")
                if not clip_path or not os.path.exists(clip_path):
                    return {"error": f"clip_path '{clip_path}' not found. Call get_audio first to obtain one."}
                with open(clip_path, "rb") as f:
                    resp = self.session.post(
                        f"{AUDIO_SERVICE_URL}/process-audio",
                        files={"file": (os.path.basename(clip_path), f, "audio/wav")},
                        timeout=60,
                    )
                resp.raise_for_status()
                data = resp.json()
                return {"cleaned_file": data.get("cleaned_file")}

            elif tool_name == "get_history":
                camera_id = tool_args.get("camera_id") or candidate_event.get("camera_id")
                minutes = tool_args.get("minutes", 60)
                since_iso = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
                records = db.get_recent_records_for_camera(camera_id, since_iso)
                return {"camera_id": camera_id, "window_minutes": minutes, "records": records}

            return {"error": f"Unknown tool '{tool_name}'"}

        except requests.RequestException as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return {"error": f"Tool execution network error: {e}"}

    def seal_and_store_verdict(self, incident_record: Dict[str, Any]):
        """
        Signs and chains the record via ledger.py, writes it via database.py.
        Locked so a watchdog-forced write and a normal write can never
        interleave mid get_last_hash()/insert_record(), which would fork the
        hash chain (this was a real, observed failure mode before the lock).
        """
        with self._ledger_lock:
            prev_h = db.get_last_hash()

            event_str = json.dumps(incident_record)
            t_now = datetime.utcnow().isoformat()

            curr_h = ledger.compute_hash(prev_h, event_str, t_now)
            sig_hex = ledger.sign_data(self.priv_key, event_str).hex()

            db.insert_record(incident_record, prev_h, curr_h, sig_hex, self.pub_key_hex, t_now)
        logger.info(f"Sealed record in ledger. Hash: {curr_h[:12]}...")

    def _finalize(self, candidate_event: Dict[str, Any], tool_call_trace: List[Dict[str, Any]],
                  action: str, args: Dict[str, Any], turn_timings: List[Dict[str, Any]],
                  episode_start: float):
        total_elapsed = time.monotonic() - episode_start
        severity = args.get("severity", "LOW" if action == "log_benign" else "HIGH")
        original_action = action

        # Guardrail: a LOW-severity "alert" is a contradiction — if the model's own
        # severity rating is LOW, nothing here actually warrants an alert. Downgrade
        # rather than trust the label, so a non-issue can't show up as RAISE_ALERT
        # on the demo dashboard just because the prompt wasn't followed perfectly.
        if action == "raise_alert" and str(severity).upper() == "LOW":
            logger.warning("Guardrail: raise_alert with LOW severity is contradictory "
                            "(a LOW-severity 'alert' isn't actionable). Downgrading to log_benign.")
            action = "log_benign"

        justification = args.get("justification") or args.get("reason")
        incident_record = {
            "camera": candidate_event.get("camera_id"),
            "tier1_triggers": candidate_event,
            "tool_call_trace": tool_call_trace,
            "action": action,
            "severity": severity,
            "justification": justification or "",
            "evidence_ids": args.get("evidence_ids", []),
            "details": args,
            "timing": {
                "total_seconds": round(total_elapsed, 2),
                "turns": turn_timings,
            },
        }
        if original_action != action:
            incident_record["downgraded_from"] = original_action
        self.seal_and_store_verdict(incident_record)

        breakdown = ", ".join(f"{t['kind']}(think={t['think']})={t['elapsed_seconds']}s" for t in turn_timings)
        logger.info(f"Episode complete in {total_elapsed:.2f}s across {len(turn_timings)} LLM turn(s): {breakdown}")

        if action == "raise_alert":
            logger.warning(f"ALERT RAISED: {justification}")
        else:
            logger.info(f"EVENT LOGGED BENIGN: {justification}")

    def investigate_event(self, candidate_event: Dict[str, Any]):
        """
        The core multi-turn loop. Always ends in exactly one raise_alert or log_benign.

        Every turn is first asked with think=False (fast, for tool selection). If
        that draft response is itself a terminal action, we don't know that until
        after the call — so we re-ask the SAME turn with think=True to get the
        deep-reasoned version that actually becomes the signed justification. The
        shallow draft is discarded, not logged into the conversation history.
        """
        logger.info("=== New Tier-1 Candidate Event Detected ===")
        episode_start = time.monotonic()

        messages = [
            {"role": "system", "content": SECURITY_MASTER_PROMPT},
            {"role": "user", "content": f"New candidate event: {json.dumps(candidate_event)}"}
        ]

        tool_call_trace: List[Dict[str, Any]] = []
        turn_timings: List[Dict[str, Any]] = []
        parse_failures = 0

        while True:
            response, elapsed = self.send_to_llm(messages, think=False)
            turn_timings.append({"kind": "intermediate", "think": False, "elapsed_seconds": round(elapsed, 2)})

            if response is None:
                parse_failures += 1
                if parse_failures > MAX_PARSE_RETRIES:
                    logger.error("LLM repeatedly failed to produce a valid response. Forcing log_benign fallback.")
                    self._finalize(candidate_event, tool_call_trace, "log_benign", {
                        "reason": "LLM failed to produce a valid tool-call/action response after retries",
                        "evidence_ids": [],
                    }, turn_timings, episode_start)
                    return
                messages.append({
                    "role": "user",
                    "content": "Your last response was not valid JSON matching the required schema. "
                                "Respond ONLY with a JSON object containing either "
                                '"tool_call"/"tool_args" or "action"/"args".',
                })
                continue

            if response.get("action") in ("raise_alert", "log_benign"):
                logger.info("Draft (think=False) terminal decision received; re-asking with think=True "
                            "for the deep-reasoned version that will be recorded.")
                deep_response, deep_elapsed = self.send_to_llm(messages, think=True)
                turn_timings.append({"kind": "final", "think": True, "elapsed_seconds": round(deep_elapsed, 2)})

                if deep_response is None:
                    logger.warning("think=True re-ask failed; falling back to the think=False draft decision.")
                    deep_response = response

                if deep_response.get("action") in ("raise_alert", "log_benign"):
                    action = deep_response["action"]
                    logger.info(f"Terminal action decided: {action.upper()}")
                    self._finalize(candidate_event, tool_call_trace, action, deep_response.get("args", {}),
                                    turn_timings, episode_start)
                    return

                # Deeper reasoning changed its mind and wants a tool instead of terminating —
                # fold it into the normal tool-call handling below.
                response = deep_response

            messages.append({"role": "assistant", "content": json.dumps(response)})

            tool_name = response.get("tool_call")
            if not tool_name:
                logger.error("LLM response lacked a valid action or tool_call. Forcing log_benign fallback.")
                self._finalize(candidate_event, tool_call_trace, "log_benign", {
                    "reason": "LLM response did not contain a valid tool_call or terminal action",
                    "evidence_ids": [],
                }, turn_timings, episode_start)
                return

            if len(tool_call_trace) >= MAX_TOOL_CALLS:
                logger.warning("Max tool calls reached without a terminal action. Forcing log_benign fallback.")
                self._finalize(candidate_event, tool_call_trace, "log_benign", {
                    "reason": "Investigation truncated after max tool calls without a terminal decision",
                    "evidence_ids": [],
                }, turn_timings, episode_start)
                return

            tool_args = response.get("tool_args", {})
            result = self.execute_tool(tool_name, tool_args, candidate_event)
            tool_call_trace.append({
                "tool": tool_name,
                "args": tool_args,
                "result": result,
                "at": datetime.utcnow().isoformat(),
            })
            messages.append({"role": "user", "content": f"Tool '{tool_name}' result: {json.dumps(result)}"})
