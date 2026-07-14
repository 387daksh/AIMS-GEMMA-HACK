import hashlib
import json
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

def compute_hash(prev_hash, event_data_str, timestamp):
    """
    Takes the previous block's fingerprint, current data, and timestamp,
    and squishes them into a unique 64-character SHA-256 string.
    """
    block_content = f"{prev_hash}{event_data_str}{timestamp}".encode('utf-8')
    return hashlib.sha256(block_content).hexdigest()

def generate_key_pair():
    """Generates a keypair for testing before Daksh hooks up the real agent."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key

def sign_data(private_key, data_str: str) -> bytes:
    """Signs data using the private key (The Wax Seal Stamp)."""
    return private_key.sign(data_str.encode('utf-8'))

def verify_signature(public_key_hex: str, signature_hex: str, data_str: str) -> bool:
    """Verifies if a specific signature matches the data using the public key."""
    try:
        pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pub_key.verify(bytes.fromhex(signature_hex), data_str.encode('utf-8'))
        return True
    except (InvalidSignature, ValueError):
        return False

def verify_chain(records) -> tuple[bool, str]:
    """
    The Auditor Loop.
    Walks through the entire database line-by-line to check if any data was altered.
    """
    if not records:
        return True, "Ledger is empty. System healthy."

    expected_prev_hash = "0" * 64
    
    for idx, record in enumerate(records):
        rec_id, timestamp, event_data, prev_hash, current_hash, signature, pub_key, _ = record
        
        # 1. Chain Continuity Check
        if prev_hash != expected_prev_hash:
            return False, f"Chain broken at Record #{rec_id}: Previous hash mismatch!"
        
        # 2. Internal Hash Check
        calculated_hash = compute_hash(prev_hash, event_data, timestamp)
        if current_hash != calculated_hash:
            return False, f"Chain broken at Record #{rec_id}: Hash integrity compromised!"
            
        # 3. Authenticity Check
        is_sig_valid = verify_signature(pub_key, signature, event_data)
        if not is_sig_valid:
            return False, f"Chain broken at Record #{rec_id}: Fraudulent signature detected!"
            
        expected_prev_hash = current_hash
        
    return True, "All cryptosignatures valid. Ledger integrity fully verified."