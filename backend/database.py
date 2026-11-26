"""
backend/database.py
-------------------
SQLite database interface for storing and retrieving deepfake scan history.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH

_db_initialized = False


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    global _db_initialized
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
            cursor = conn.execute("PRAGMA table_info(scan_history)")
            columns = {row[1] for row in cursor.fetchall()}
            if "heatmap_base64" not in columns:
                conn.execute("ALTER TABLE scan_history ADD COLUMN heatmap_base64 TEXT")
            if "forensic_data" not in columns:
                conn.execute("ALTER TABLE scan_history ADD COLUMN forensic_data TEXT")
        _db_initialized = True
    finally:
        conn.close()


def _ensure_db() -> None:
    if not _db_initialized:
        init_db()


def _parse_json_field(raw: Optional[str], default: Any, field_name: str) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[Database Warning] Corrupted JSON in '{field_name}': {exc}")
        return default


def _row_to_scan_dict(row: sqlite3.Row) -> Dict[str, Any]:
    keys = row.keys()
    flags = _parse_json_field(row["metadata_flags"], [], "metadata_flags")
    forensic_dict = _parse_json_field(
        row["forensic_data"] if "forensic_data" in keys else None,
        {},
        "forensic_data",
    )
    heatmap_b64 = row["heatmap_base64"] if "heatmap_base64" in keys else None

    return {
        "id": row["id"],
        "report_id": row["report_id"],
        "filename": row["filename"],
        "verdict": row["verdict"],
        "fake_probability": row["fake_probability"],
        "face_detected": bool(row["face_detected"]),
        "total_faces": row["total_faces"],
        "metadata_flags": flags,
        "ela_flag": bool(row["ela_flag"]),
        "model_name": row["model_name"],
        "processing_time_ms": row["processing_time_ms"],
        "created_at": row["created_at"],
        "heatmap_base64": heatmap_b64,
        "faces": forensic_dict.get("faces", []),
        "regional_influence": forensic_dict.get("regional_influence", []),
        "forensic_rationale": forensic_dict.get("forensic_rationale", ""),
    }


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
    forensic_rationale: Optional[str] = None,
    faces: Optional[List[Dict[str, Any]]] = None,
) -> None:
    _ensure_db()
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    flags_json = json.dumps(metadata_flags)
    forensic_json = json.dumps({
        "faces": faces or [],
        "regional_influence": regional_influence or [],
        "forensic_rationale": forensic_rationale or "",
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
                forensic_json,
            ))
    finally:
        conn.close()


def get_all_scans(limit: int = 50) -> List[Dict[str, Any]]:
    _ensure_db()
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM scan_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_scan_dict(r) for r in rows]


def get_scan_by_report_id(report_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM scan_history WHERE report_id = ? LIMIT 1",
            (report_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_scan_dict(row) if row else None
