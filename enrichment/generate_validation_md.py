"""Dynamic validation chain Markdown generator for the Family Office pipeline.

Reads ``family_offices.csv``, ``candidates.csv``, and ``provenance_log.csv``,
programmatically selects the 3 highest-quality records, investigates their full
history, and writes ``validation_chain.md`` to the project root.

Usage::

    python enrichment/generate_validation_md.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------

_BASE_DIR: str = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR: str = os.path.join(_BASE_DIR, "data")

FAMILIES_PATH: str = os.path.join(DATA_DIR, "family_offices.csv")
CANDIDATES_PATH: str = os.path.join(DATA_DIR, "candidates.csv")
PROVENANCE_PATH: str = os.path.join(DATA_DIR, "provenance_log.csv")
OUTPUT_PATH: str = os.path.join(_BASE_DIR, "validation_chain.md")

# Fields we want to backfill from provenance_log
PROVENANCE_FIELDS: list[str] = [
    "principal_name",
    "principal_title",
    "recent_signal",
    "website",
    "entity_linkedin",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_str(val: object) -> str:
    """Coerce *val* to a trimmed string, returning ``""`` on falsy/NaN."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def _lookup_discovery_url(
    entity_name: str,
    existing_url: str,
    df_candidates: pd.DataFrame,
) -> str:
    """Return the best discovery URL for *entity_name*.

    If *existing_url* is valid (not empty and not "Could not verify"),
    it is returned as-is. Otherwise the candidates CSV is searched for a
    matching ``entity_name`` row and its ``source_url`` is returned.

    Args:
        entity_name: The entity to look up.
        existing_url: The ``source_url`` from ``family_offices.csv``.
        df_candidates: The candidates DataFrame.

    Returns:
        A URL string or ``"Could not verify"``.
    """
    if existing_url and existing_url.lower() not in ("could not verify", "none", "null"):
        return existing_url

    match = df_candidates[df_candidates["entity_name"] == entity_name]
    if match.empty:
        return "Could not verify"

    url = _safe_str(match.iloc[0].get("source_url", ""))
    return url if url else "Could not verify"


def _lookup_provenance_url(
    entity_name: str,
    field_name: str,
    df_provenance: pd.DataFrame,
) -> str:
    """Return the first provenance ``source_url`` for a given field.

    Args:
        entity_name: The entity to look up.
        field_name: The target column name (e.g. ``"principal_name"``).
        df_provenance: The provenance log DataFrame.

    Returns:
        The first matching URL string, or ``"Could not verify"``.
    """
    mask = (df_provenance["entity_name"] == entity_name) & (
        df_provenance["field_name"] == field_name
    )
    matches = df_provenance[mask]
    if matches.empty:
        return "Could not verify"

    url = _safe_str(matches.iloc[0].get("source_url", ""))
    return url if url else "Could not verify"


def _extraction_method(source: str) -> str:
    """Return a human-readable extraction-method sentence for *source*.

    Args:
        source: The ``discovery_source`` value.

    Returns:
        A one-sentence description.
    """
    mapping: dict[str, str] = {
        "SEC EDGAR": (
            "SEC EDGAR EFTS Full-Text Search parsed the primary_doc.xml "
            "filing to extract entity metadata."
        ),
        "IAPD": (
            "IAPD Investment Adviser Search API retrieved firm details "
            "via the CRD number."
        ),
        "ProPublica": (
            "ProPublica Nonprofit Explorer API was used with beacon "
            "conversion to identify the operating entity."
        ),
    }
    return mapping.get(source, f"Discovered via {source}.")


def _deterministic_fetch(source: str) -> str:
    """Return a deterministic-fetch description based on *source*.

    Args:
        source: The ``discovery_source`` value.

    Returns:
        A one-sentence description.
    """
    mapping: dict[str, str] = {
        "SEC EDGAR": (
            "SEC EDGAR primary_doc.xml was parsed to verify principal "
            "name and filing metadata."
        ),
        "IAPD": (
            "IAPD Form ADV JSON was used to verify principal name, "
            "client count, and AUM range."
        ),
        "ProPublica": "Deferred to LLM",
    }
    return mapping.get(source, "Deferred to LLM")


def _confidence_sentence(status: str, source: str) -> str:
    """Return a brief confidence explanation.

    Args:
        status: The ``verification_status`` value (High / Medium / Low).
        source: The ``discovery_source`` value.

    Returns:
        A one-sentence string.
    """
    if source in ("SEC EDGAR", "IAPD"):
        return (
            "Confidence is elevated because structured XML/JSON grounding "
            "supplemented the LLM enrichment."
        )
    return "Confidence is based solely on LLM extraction of web search snippets."


# ---------------------------------------------------------------------------
# Selection Logic
# ---------------------------------------------------------------------------


def _select_top3(
    df_families: pd.DataFrame,
) -> pd.DataFrame:
    """Select the top 3 records for the validation chain.

    Prioritises ``verification_status`` == "High" then "Medium",
    ``entity_type`` == "SFO" over "SFO (Probable)", and attempts
    to pick three distinct ``discovery_source`` values.

    Args:
        df_families: The family offices DataFrame.

    Returns:
        A DataFrame containing exactly 3 selected rows.
    """
    df = df_families.copy()

    # --- Status ordering ---
    status_order: dict[str, int] = {"High": 0, "Medium": 1, "Low": 2}
    df["_status_rank"] = df["verification_status"].map(
        lambda x: status_order.get(str(x).strip(), 99)
    )

    # --- Entity-type ordering (SFO before Probable) ---
    df["_type_rank"] = df["entity_type"].map(
        lambda x: 0 if str(x).strip() == "SFO" else 1
    )

    # Sort by: status, type, then entity name for determinism
    df = df.sort_values(
        by=["_status_rank", "_type_rank", "entity_name"],
        ascending=True,
    )

    # --- Greedy selection with source diversity ---
    selected: list[pd.Series] = []
    seen_sources: set[str] = set()

    for _, row in df.iterrows():
        if len(selected) >= 3:
            break
        source = _safe_str(row.get("discovery_source", ""))
        # Prefer a source we haven't seen yet, but if we already have 2 and
        # no new source is available, just pick the next best.
        if source not in seen_sources or len(selected) >= 2:
            selected.append(row)
            seen_sources.add(source)

    # Safety: if somehow we have fewer than 3 (shouldn't happen), fill from top
    if len(selected) < 3:
        for _, row in df.iterrows():
            if len(selected) >= 3:
                break
            if row.name not in {r.name for r in selected}:
                selected.append(row)

    result = pd.DataFrame(selected[:3])
    result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Markdown Generation
# ---------------------------------------------------------------------------


def _build_md(
    selected: pd.DataFrame,
    df_candidates: pd.DataFrame,
    df_provenance: pd.DataFrame,
) -> str:
    """Build the complete validation_chain.md content.

    Args:
        selected: The 3 selected records.
        df_candidates: The candidates DataFrame.
        df_provenance: The provenance log DataFrame.

    Returns:
        The full Markdown string.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    lines: list[str] = [
        "# Task 1: 3-Record Validation Chain",
        "",
        f"**Report Generated:** {today}",
        "",
        (
            "The following three records demonstrate the full "
            "discovery-to-validation lifecycle of the pipeline. They were "
            "programmatically selected to showcase the system's ability to "
            "handle different discovery sources and apply deterministic "
            "structured grounding alongside LLM-based web enrichment."
        ),
        "",
    ]

    for idx, (_, row) in enumerate(selected.iterrows(), start=1):
        entity_name = _safe_str(row.get("entity_name", "Unknown"))
        source = _safe_str(row.get("discovery_source", "Unknown"))
        status = _safe_str(row.get("verification_status", "Low"))
        notes = _safe_str(row.get("verification_notes", "None"))
        existing_url = _safe_str(row.get("source_url", ""))

        # Backfill
        discovery_url = _lookup_discovery_url(
            entity_name, existing_url, df_candidates
        )
        prov_urls: dict[str, str] = {}
        for field in PROVENANCE_FIELDS:
            prov_urls[field] = _lookup_provenance_url(
                entity_name, field, df_provenance
            )

        # Build record section
        lines.append("---")
        lines.append("")
        lines.append(f"## Record {idx}: {entity_name}")
        lines.append("")
        lines.append(f"**1. Discovery Source:**  ")
        lines.append(f"{source}")
        lines.append("")
        lines.append(f"**2. Extraction Method:**  ")
        lines.append(_extraction_method(source))
        lines.append("")
        lines.append(f"**3. Enrichment Steps:**  ")
        lines.append(f"- **Deterministic Fetch:** {_deterministic_fetch(source)}")
        lines.append(
            "- **Web Search:** Queried Bing for principal LinkedIn and "
            "recent news/signals."
        )
        lines.append(
            "- **Pre-Filter:** RapidFuzz stripped generic tokens to "
            "validate snippet relevance."
        )
        lines.append(
            "- **LLM Extraction:** Groq LLM processed anchored snippets "
            "to extract web data."
        )
        lines.append("")
        lines.append(f"**4. Validation Logic:**  ")
        lines.append(
            "- **Grounding:** The pipeline used ID-based grounding to map "
            "LLM extractions back to real Bing search result URLs."
        )
        lines.append(
            "- **MFO Filter:** The pipeline confirmed the firm was not "
            "classified as an MFO."
        )
        if notes and notes.lower() not in ("none", "could not verify"):
            lines.append(f"- {notes}")
        lines.append("")
        lines.append(f"**5. Confidence Assessment:**  ")
        lines.append(f"**{status}**. {_confidence_sentence(status, source)}")
        lines.append("")
        lines.append(f"**6. Exact Sources or Links Used:**  ")
        lines.append(f"- **Discovery Source URL:** {discovery_url}")
        lines.append(
            f"- **Principal Name Source URL:** {prov_urls['principal_name']}"
        )
        lines.append(
            f"- **Recent Signal Source URL:** {prov_urls['recent_signal']}"
        )
        lines.append(
            f"- **Website/LinkedIn Source URL:** "
            f"{prov_urls['website']} / {prov_urls['entity_linkedin']}"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate() -> None:
    """Entry point: load data, select records, write Markdown."""
    # --- Load data ---
    try:
        df_families = pd.read_csv(FAMILIES_PATH, keep_default_na=False)
    except FileNotFoundError:
        print(f"ERROR: {FAMILIES_PATH} not found. Run enrichment first.")
        raise SystemExit(1)

    try:
        df_candidates = pd.read_csv(CANDIDATES_PATH, keep_default_na=False)
    except FileNotFoundError:
        print(f"WARNING: {CANDIDATES_PATH} not found. Discovery URL backfill disabled.")
        df_candidates = pd.DataFrame()

    try:
        df_provenance = pd.read_csv(PROVENANCE_PATH, keep_default_na=False)
    except FileNotFoundError:
        print(f"WARNING: {PROVENANCE_PATH} not found. Provenance URL backfill disabled.")
        df_provenance = pd.DataFrame()

    # --- Select top 3 ---
    selected = _select_top3(df_families)
    print(f"Selected {len(selected)} records for validation chain.")

    # --- Build & write Markdown ---
    md_content = _build_md(selected, df_candidates, df_provenance)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(md_content)

    print(f"validation_chain.md written to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
