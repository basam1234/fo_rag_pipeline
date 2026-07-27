"""
Normalization, CSV persistence, and provenance logging for the discovery pipeline.

Responsibilities:
- Define the Candidate dataclass
- Normalize raw API records into Candidate instances
- Append candidates to data/candidates.csv
- Append provenance entries to data/provenance_log.csv
- Log per-record errors to data/pipeline_errors.csv

This module is intentionally independent of all fetcher modules.
"""
import json
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")


def _ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Candidate:
    """A single normalized candidate entity from any discovery source."""

    entity_name: Optional[str]
    principal_name: Optional[str]
    address: Optional[str]
    discovery_source: str
    source_url: str
    entity_type: Optional[str]
    aum_range: Optional[str]
    raw_payload: str


def normalize_record(raw_data: dict, source_type: str, source_url: str) -> Candidate:
    """Safely extract fields from a raw API record into a Candidate dataclass.

    Missing or empty values map to ``None``, which writes as an empty cell in CSV.

    Args:
        raw_data: The raw JSON dict from the API response.
        source_type: The discovery source label (e.g. "SEC EDGAR").
        source_url: The canonical URL for this record.

    Returns:
        A fully populated Candidate instance.
    """
    entity_name = _safe_get(raw_data, "entity_name")
    principal_name = _safe_get(raw_data, "principal_name")
    address = _safe_get(raw_data, "address")
    entity_type = _safe_get(raw_data, "entity_type")
    aum_range = _safe_get(raw_data, "aum_range")

    return Candidate(
        entity_name=entity_name,
        principal_name=principal_name,
        address=address,
        discovery_source=source_type,
        source_url=source_url,
        entity_type=entity_type,
        aum_range=aum_range,
        raw_payload=json.dumps(raw_data, default=str),
    )


def _safe_get(data: dict, key: str) -> Optional[str]:
    """Return the value for *key* if it exists and is non-empty, else None."""
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


_CANDIDATES_CSV = DATA_DIR / "candidates.csv"
_CANDIDATE_HEADERS = [
    "entity_name",
    "principal_name",
    "address",
    "discovery_source",
    "source_url",
    "entity_type",
    "aum_range",
    "raw_payload",
]


def write_candidate(candidate: Candidate) -> None:
    """Append a single candidate to ``data/candidates.csv``.

    If the file does not exist, it is created with the required header row.
    """
    _ensure_data_dir()
    file_exists = _CANDIDATES_CSV.exists()
    with open(_CANDIDATES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CANDIDATE_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(candidate))


_PROVENANCE_CSV = DATA_DIR / "provenance_log.csv"
_PROVENANCE_HEADERS = [
    "entity_name",
    "field_name",
    "source_type",
    "source_url",
    "verification_method",
    "confidence",
    "date",
]


def write_provenance(
    entity_name: Optional[str],
    field_name: str,
    source_type: str,
    source_url: str,
    verification_method: str,
    confidence: str,
) -> None:
    """Append a provenance log entry to ``data/provenance_log.csv``.

    If the file does not exist, it is created with the required header row.
    """
    _ensure_data_dir()
    file_exists = _PROVENANCE_CSV.exists()
    with open(_PROVENANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROVENANCE_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "entity_name": entity_name or "",
                "field_name": field_name,
                "source_type": source_type,
                "source_url": source_url,
                "verification_method": verification_method,
                "confidence": confidence,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        )


_ERROR_LOG = DATA_DIR / "pipeline_errors.csv"
_ERROR_HEADERS = ["source", "entity_name", "error_message", "timestamp"]


def log_error(source: str, entity_name: Optional[str], error_message: str) -> None:
    """Append an error record to ``data/pipeline_errors.csv``."""
    _ensure_data_dir()
    file_exists = _ERROR_LOG.exists()
    with open(_ERROR_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ERROR_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "source": source,
                "entity_name": entity_name or "",
                "error_message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def save_candidate(candidate: Candidate) -> None:
    """Persist a candidate and log provenance for its three high-value fields.

    After writing the candidate row, three provenance records are written
    for Entity Name, Principal Name, and Address using the same source
    metadata and ``verification_method="Primary Legal Filing"`` /
    ``confidence="High"``.
    """
    write_candidate(candidate)
    write_provenance(
        entity_name=candidate.entity_name,
        field_name="Entity Name",
        source_type=candidate.discovery_source,
        source_url=candidate.source_url,
        verification_method="Primary Legal Filing",
        confidence="High",
    )
    write_provenance(
        entity_name=candidate.entity_name,
        field_name="Principal Name",
        source_type=candidate.discovery_source,
        source_url=candidate.source_url,
        verification_method="Primary Legal Filing",
        confidence="High",
    )
    write_provenance(
        entity_name=candidate.entity_name,
        field_name="Address",
        source_type=candidate.discovery_source,
        source_url=candidate.source_url,
        verification_method="Primary Legal Filing",
        confidence="High",
    )
