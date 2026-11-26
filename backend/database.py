"""
backend/database.py
-------------------
SQLite database interface for storing and retrieving deepfake scan history.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scans.db"


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite scan history database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes the scan history database schema if not present, and handles migrations."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    fake_probability REAL NOT NULL,
                    face_detected INTEGER NOT NULL,
                    total_faces INTEGER NOT NULL,
                    metadata_flags TEXT,
                    ela_flag INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    processing_time_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    heatmap_base64 TEXT,
                    forensic_data TEXT
                )
            """)
            # Check existing columns and migrate if necessary
            cursor = conn.execute("PRAGMA table_info(scan_history)")
            columns = [row[1] for row in cursor.fetchall()]
            if "heatmap_base64" not in columns:
                conn.execute("ALTER TABLE scan_history ADD COLUMN heatmap_base64 TEXT")
            if "forensic_data" not in columns:
                conn.execute("ALTER TABLE scan_history ADD COLUMN forensic_data TEXT")
    finally:
        conn.close()


def insert_scan(
    report_id: str,
    filename: str,
    verdict: str,
    fake_probability: float,
    face_detected: bool,
    total_faces: int,
    metadata_flags: List[str],
    ela_flag: bool,
    model_name: str,
    processing_time_ms: float,
    created_at: Optional[str] = None,
    heatmap_base64: Optional[str] = None,
    regional_influence: Optional[List[Dict[str, Any]]] = None,
    forensic_rationale: Optional[str] = None
) -> None:
    """Inserts a completed scan into the database."""
    init_db()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    flags_json = json.dumps(metadata_flags)
    forensic_json = json.dumps({
        "regional_influence": regional_influence or [],
        "forensic_rationale": forensic_rationale or ""
    })
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO scan_history (
                    report_id, filename, verdict, fake_probability,
                    face_detected, total_faces, metadata_flags,
                    ela_flag, model_name, processing_time_ms, created_at,
                    heatmap_base64, forensic_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                filename,
                verdict,
                fake_probability,
                1 if face_detected else 0,
                total_faces,
                flags_json,
                1 if ela_flag else 0,
                model_name,
                round(processing_time_ms, 2),
                created_at,
                heatmap_base64,
                forensic_json
            ))
    finally:
        conn.close()


def get_all_scans(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent scans ordered by creation date descending."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM scan_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    results = []
    for r in rows:
        flags = []
        if r["metadata_flags"]:
            try:
                flags = json.loads(r["metadata_flags"])
            except Exception:
                flags = []

        keys = r.keys()
        h_b64 = r["heatmap_base64"] if "heatmap_base64" in keys else None
        forensic_dict = {}
        if "forensic_data" in keys and r["forensic_data"]:
            try:
                forensic_dict = json.loads(r["forensic_data"])
            except Exception:
                forensic_dict = {}

        results.append({
            "id": r["id"],
            "report_id": r["report_id"],
            "filename": r["filename"],
            "verdict": r["verdict"],
            "fake_probability": r["fake_probability"],
            "face_detected": bool(r["face_detected"]),
            "total_faces": r["total_faces"],
            "metadata_flags": flags,
            "ela_flag": bool(r["ela_flag"]),
            "model_name": r["model_name"],
            "processing_time_ms": r["processing_time_ms"],
            "created_at": r["created_at"],
            "heatmap_base64": h_b64,
            "gradcam_url": h_b64,
            "regional_influence": forensic_dict.get("regional_influence", []),
            "forensic_rationale": forensic_dict.get("forensic_rationale", "")
        })
    return results


def get_scan_by_report_id(report_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific scan by report_id."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.execute("""
            SELECT * FROM scan_history
            WHERE report_id = ?
            LIMIT 1
        """, (report_id,))
        r = cursor.fetchone()
    finally:
        conn.close()

    if not r:
        return None

    flags = []
    if r["metadata_flags"]:
        try:
            flags = json.loads(r["metadata_flags"])
        except Exception:
            flags = []

    keys = r.keys()
    h_b64 = r["heatmap_base64"] if "heatmap_base64" in keys else None
    forensic_dict = {}
    if "forensic_data" in keys and r["forensic_data"]:
        try:
            forensic_dict = json.loads(r["forensic_data"])
        except Exception:
            forensic_dict = {}

    return {
        "id": r["id"],
        "report_id": r["report_id"],
        "filename": r["filename"],
        "verdict": r["verdict"],
        "fake_probability": r["fake_probability"],
        "face_detected": bool(r["face_detected"]),
        "total_faces": r["total_faces"],
        "metadata_flags": flags,
        "ela_flag": bool(r["ela_flag"]),
        "model_name": r["model_name"],
        "processing_time_ms": r["processing_time_ms"],
        "created_at": r["created_at"],
        "heatmap_base64": h_b64,
        "gradcam_url": h_b64,
        "regional_influence": forensic_dict.get("regional_influence", []),
        "forensic_rationale": forensic_dict.get("forensic_rationale", "")
    }
