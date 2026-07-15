import sqlite3
import json

# This is the name of the file where our database spreadsheet will live on your Mac
DB_NAME = "ledger.db"

def init_db():
    """
    Step 1A: Initialize the Database.
    This sets up our columns and tables so they match the team contract perfectly.
    """
    # Connect to the database file (it creates the file automatically if it's missing)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # We create the 'spreadsheet' table with the exact columns we agreed upon
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incident_record (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_data TEXT NOT NULL,      -- Stored as a JSON string (Gemma's output)
            previous_hash TEXT NOT NULL,   -- Fingerprint of the block before this one
            current_hash TEXT NOT NULL,    -- Fingerprint of this specific block
            signature TEXT NOT NULL,       -- Hex-encoded Ed25519 signature from Gemma
            public_key TEXT NOT NULL,      -- Hex-encoded Ed25519 public key
            is_suppressed INTEGER DEFAULT 0 -- 0 = Visible, 1 = Hidden (For our Phase 4 UI toggle)
        )
    """)
    
    # Save our changes and close the connection safely
    conn.commit()
    conn.close()

def insert_record(event_data_dict, prev_hash, curr_hash, signature_hex, pub_key_hex, timestamp):
    """
    Step 1B: Insert a Record.
    This function lets Daksh's agent loop push new logs into our database easily.

    timestamp must be the exact same value the caller used to compute curr_hash
    (via ledger.compute_hash). Generating a fresh timestamp here instead of
    accepting the caller's would desync the stored row from the hash that was
    actually signed, making every record fail verification.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # json.dumps converts a Python dictionary {} into a clean text string so the DB can store it
    cursor.execute("""
        INSERT INTO incident_record (timestamp, event_data, previous_hash, current_hash, signature, public_key)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, json.dumps(event_data_dict), prev_hash, curr_hash, signature_hex, pub_key_hex))
    
    conn.commit()
    conn.close()

def get_all_records(include_suppressed=False):
    """
    Step 1C: Fetch Records.
    This reads the database so your Streamlit UI dashboard can display the logs.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # If include_suppressed is True, show everything. If False, filter out the hidden/benign stuff.
    if include_suppressed:
        cursor.execute("SELECT * FROM incident_record ORDER BY id ASC")
    else:
        cursor.execute("SELECT * FROM incident_record WHERE is_suppressed = 0 ORDER BY id ASC")
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def toggle_suppression(record_id, suppress_status):
    """
    Step 1D: Hide/Show an Event.
    Allows us to click a button on the dashboard to hide benign alerts without deleting them.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE incident_record SET is_suppressed = ? WHERE id = ?", (1 if suppress_status else 0, record_id))
    conn.commit()
    conn.close()

def get_recent_records_for_camera(camera_id, since_iso, include_suppressed=True):
    """
    Step 1E: Camera History.
    Filters the full ledger down to records for one camera within a time
    window, for the orchestrator's get_history tool. Reuses get_all_records
    rather than raw SQL JSON extraction, since event_data is stored as a
    JSON string and we want this portable across SQLite builds.
    """
    records = get_all_records(include_suppressed=include_suppressed)
    matches = []

    for record in records:
        rec_id, timestamp, event_data_str, _, _, _, _, is_suppressed = record

        if timestamp < since_iso:
            continue

        try:
            event = json.loads(event_data_str)
        except (json.JSONDecodeError, TypeError):
            continue

        record_camera = event.get("camera") or event.get("tier1_triggers", {}).get("camera_id")
        if record_camera != camera_id:
            continue

        matches.append({
            "id": rec_id,
            "timestamp": timestamp,
            "severity": event.get("severity", "UNKNOWN"),
            "justification": event.get("justification", ""),
            "action": event.get("action", ""),
            "is_suppressed": bool(is_suppressed),
        })

    return matches


# This part only runs if YOU execute this file directly to test it!
def get_last_hash():
    """
    Looks at the very last block in the database
    so the next block can link to it mathematically.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT current_hash FROM incident_record ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    else:
        # If the database is completely empty, start with a default block of 64 zeros
        return "0" * 64

# Keep this at the absolute bottom of the file!
if __name__ == "__main__":
    init_db()
    print("🚀 Database initialized and ledger table successfully created!")