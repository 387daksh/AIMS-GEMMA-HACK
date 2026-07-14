import streamlit as st
import database as db
import ledger
import json

# Set up page configurations
st.set_page_config(page_title="Gemma Security Ledger", layout="wide")
db.init_db()

st.title("🛡️ Gemma AI Agent: Security Ledger & Audit Dashboard")
st.markdown("Real-time cryptographic proof-of-incident tracking dashboard panel.")

# --- SIDEBAR INTERFACE CONTROLS ---
st.sidebar.header("Control Panel Options")
show_suppressed = st.sidebar.checkbox("Show Suppressed / Benign Incident Alerts", value=False)

# Pull current dataset rows out of SQLite database
records = db.get_all_records(include_suppressed=show_suppressed)

# --- LEDGER CRYPTO INTEGRITY STATUS BLOCK ---
st.subheader("Automated Cryptographic Integrity Audit")
is_valid, message = ledger.verify_chain(db.get_all_records(include_suppressed=True))

if is_valid:
    st.success(f"💚 **LEDGER SECURE:** {message}")
else:
    st.error(f"🚨 **WARNING - CRYPTOGRAPHIC INTEGRITY COMPROMISED:** {message}")

# --- METRIC SUMMARY WIDGETS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Logged Incidents", value=len(records))
with col2:
    st.metric(label="Chain Validation State", value="VERIFIED" if is_valid else "TAMPERED")
with col3:
    st.metric(label="Agent Defenses", value="OPERATIONAL" if is_valid else "LOCKDOWN")

st.markdown("---")

# --- TRACE PANEL ALERT LOG CARDS ---
st.subheader("Live Ledger Trace Panel")

if not records:
    st.info("Waiting for data logs from Gemma agent stream...")
else:
    for record in records:
        rec_id, timestamp, event_data_str, prev_hash, current_hash, signature, pub_key, is_suppressed = record
        
        try:
            event_json = json.loads(event_data_str)
            severity = event_json.get("severity", "LOW").upper()
        except Exception:
            severity = "LOW"
            event_json = {"raw_payload": event_data_str}

        # Select custom border styles dynamically based on the threat levels
        if severity in ["HIGH", "CRITICAL"]:
            card_style = "border-left: 8px solid #FF4B4B; background-color: #2e1a1a;"
        elif severity == "MEDIUM":
            card_style = "border-left: 8px solid #FFA500; background-color: #2e261a;"
        else:
            card_style = "border-left: 8px solid #28A745; background-color: #1a2e1f;"

        # Render the custom visual alert block card using HTML container styles
        st.markdown(f"""
        <div style="padding: 18px; border-radius: 6px; margin-bottom: 12px; {card_style}">
            <h4 style="margin: 0 0 10px 0;">Incident Block #{rec_id} | Time: {timestamp} <span style="float:right; color:#ddd;">Severity: {severity}</span></h4>
            <pre style="margin: 0 0 10px 0; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">{json.dumps(event_json, indent=2)}</pre>
            <small style="font-family: monospace; display:block; color:#aaa;"><b>Prev Hash:</b> {prev_hash}</small>
            <small style="font-family: monospace; display:block; color:#aaa;"><b>Curr Hash:</b> {current_hash}</small>
        </div>
        """, unsafe_allow_html=True)

        # Build Phase 4 Action Toggle Button for handling benign suppression
        button_label = "Unsuppress Event Log" if is_suppressed else "Suppress Incident Log"
        if st.button(button_label, key=f"btn_{rec_id}"):
            db.toggle_suppression(rec_id, not is_suppressed)
            st.rerun()

st.markdown("---")

# --- SANDBOX BLOCK INJECTOR TOOL FOR DEMO ---
with st.expander("🛠️ Interactive Testing Sandbox (Inject Mock Incident)"):
    if st.button("Generate & Inject Test Data Block"):
        # Create arbitrary key set to manufacture matching hash profiles
        priv_k, pub_k = ledger.generate_key_pair()
        pub_hex = pub_k.public_bytes_raw().hex()
        
        mock_payload = {"event": "Anomalous System Access Detected", "severity": "HIGH", "source": "Internal Terminal"}
        mock_str = json.dumps(mock_payload)
        
        sig_hex = ledger.sign_data(priv_k, mock_str).hex()
        
        all_recs = db.get_all_records(include_suppressed=True)
        prev_h = all_recs[-1][4] if all_recs else "0" * 64
        
        import datetime
        t_now = datetime.datetime.utcnow().isoformat()
        curr_h = ledger.compute_hash(prev_h, mock_str, t_now)
        
        conn = db.sqlite3.connect(db.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incident_record (timestamp, event_data, previous_hash, current_hash, signature, public_key)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (t_now, mock_str, prev_h, curr_h, sig_hex, pub_hex))
        conn.commit()
        conn.close()
        st.success("Mock cryptographic packet successfully appended to local blockchain registry!")
        st.rerun()