"""Orchestrator for the enrichment phase of the Family Office data pipeline.

Executes ordered phases:
1. Checkpoint reading (skip already-processed entities).
2. Pre-filter (EDGAR real-estate drops, IAPD inactive drops).
3. Structured data fetch (SEC EDGAR primary_doc.xml / IAPD Form ADV XML).
4. Relevance pre-filter via RapidFuzz on snippet text.
5. ID assignment to surviving snippets.
6. Groq LLM extraction with anchored prompt and ID-based grounding.
7. Merge structured XML data with LLM data.
8. 3-tier ranking and top-50 selection.
9. Final CSV persistence (``family_offices.csv``) and provenance logging.

Usage::

    python enrichment/run_enrichment.py
"""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from rapidfuzz import fuzz

from enrichment.ddg_scraper import search_ddg
from enrichment.llm_extractor import EnrichedData, extract_data_from_snippets
from enrichment.structured_fetcher import fetch_structured

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")
CANDIDATES_PATH: str = os.path.join(DATA_DIR, "candidates.csv")
CHECKPOINT_PATH: str = os.path.join(DATA_DIR, "checkpoint_enrichment.csv")
REJECTED_PATH: str = os.path.join(DATA_DIR, "enrichment_rejected_candidates.csv")
OUTPUT_PATH: str = os.path.join(DATA_DIR, "family_offices.csv")
PROVENANCE_PATH: str = os.path.join(DATA_DIR, "provenance_log.csv")

# EDGAR real-estate filter regex
RE_EDGAR_REAL_ESTATE: re.Pattern[str] = re.compile(
    r"(real estate|realty|properties)", re.IGNORECASE
)
RE_FAMILY: re.Pattern[str] = re.compile(r"family", re.IGNORECASE)

# Generic tokens to strip for RapidFuzz distilling
GENERIC_TOKENS: list[str] = [
    "LLC",
    "LP",
    "Inc",
    "Family",
    "Capital",
    "Holdings",
    "Fund",
    "Wealth",
    "Office",
    "Trust",
    "Corp",
    "Co",
]
_GENERIC_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in GENERIC_TOKENS) + r")\b",
    re.IGNORECASE,
)

# IAPD client count for hard-drop threshold
MAX_CLIENT_COUNT: int = 8

# RapidFuzz relevance threshold
RELEVANCE_THRESHOLD: float = 75.0

# Source priority for tie-breaking
SOURCE_PRIORITY: dict[str, int] = {
    "ProPublica": 0,
    "IAPD": 1,
    "SEC EDGAR": 2,
}

# Final output columns (ordered)
OUTPUT_COLUMNS: list[str] = [
    "entity_name",
    "entity_type",
    "principal_name",
    "principal_title",
    "principal_linkedin",
    "principal_email",
    "entity_linkedin",
    "website",
    "address",
    "aum_range",
    "recent_signal",
    "signal_date",
    "discovery_source",
    "source_url",
    "verification_status",
    "verification_notes",
]

# Fields considered "high-value" for confidence scoring
HIGH_VALUE_FIELDS: list[str] = [
    "principal_name",
    "principal_linkedin",
    "principal_email",
    "recent_signal",
    "website",
    "entity_linkedin",
]

# Fields sourced from the deterministic structured layer
STRUCTURED_FIELDS: set[str] = {
    "principal_name",
    "principal_title",
    "aum_range",
    "iapd_client_count",
}

# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint() -> set[str]:
    """Return a set of already-processed entity names from the checkpoint.

    Returns:
        Currently always returns an empty set (checkpoint is not
        enabled by default).
    """
    return set()


def _write_checkpoint(entity_name: str) -> None:
    """Append a single entity name and timestamp to the checkpoint CSV.

    Args:
        entity_name: The entity that was just enriched.
    """
    now = datetime.now(timezone.utc).isoformat()
    file_exists = os.path.isfile(CHECKPOINT_PATH)
    with open(CHECKPOINT_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["entity_name", "enriched_at"])
        writer.writerow([entity_name, now])


# ---------------------------------------------------------------------------
# Pre-Filter Helpers
# ---------------------------------------------------------------------------


def _is_edgar_real_estate_drop(entity_name: str) -> bool:
    """Check whether an EDGAR candidate should be dropped as real estate.

    Drops candidates whose entity_name matches real-estate keywords
    UNLESS the word "family" also appears.

    Args:
        entity_name: The candidate entity name.

    Returns:
        True if the candidate should be dropped.
    """
    if not isinstance(entity_name, str) or not entity_name:
        return False
    if RE_FAMILY.search(entity_name):
        return False
    return bool(RE_EDGAR_REAL_ESTATE.search(entity_name))


def _init_rejected_csv() -> None:
    """Create (or overwrite) the rejected-candidates CSV with a header row."""
    with open(REJECTED_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entity_name", "discovery_source", "rejection_reason"])


def _log_rejection(entity_name: str, discovery_source: str, reason: str) -> None:
    """Append a single rejection row to the rejected-candidates CSV.

    Args:
        entity_name: The candidate entity name.
        discovery_source: The discovery source string.
        reason: Human-readable rejection reason.
    """
    with open(REJECTED_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([entity_name, discovery_source, reason])


# ---------------------------------------------------------------------------
# RapidFuzz Relevance Pre-Filter
# ---------------------------------------------------------------------------


def _distill_entity_name(entity_name: str) -> str:
    """Strip generic tokens from an entity name to leave a distinctive keyword.

    Args:
        entity_name: The full entity name string.

    Returns:
        The distilled keyword with generic tokens removed and extra
        whitespace collapsed.
    """
    distilled = _GENERIC_PATTERN.sub("", entity_name)
    distilled = re.sub(r"\s{2,}", " ", distilled).strip()
    return distilled if distilled else entity_name.strip()


def _relevance_filter(
    entity_name: str,
    snippets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter snippets by RapidFuzz partial-ratio relevance to the entity.

    For each snippet, computes ``partial_ratio`` between the distilled
    entity name and the combined snippet title + text.  Drops snippets
    whose score is below ``RELEVANCE_THRESHOLD``.

    Args:
        entity_name: The candidate entity name.
        snippets: The raw snippet list from the scraper.

    Returns:
        The filtered snippet list (may be empty).
    """
    if not entity_name or not snippets:
        return snippets

    distilled = _distill_entity_name(entity_name).lower()
    if not distilled:
        return snippets

    surviving: list[dict[str, Any]] = []
    for snip in snippets:
        combined = (
            f"{snip.get('title', '')} "
            f"{snip.get('snippet', '')}"
        ).lower()
        score = fuzz.partial_ratio(distilled, combined)
        if score >= RELEVANCE_THRESHOLD:
            surviving.append(snip)

    return surviving


# ---------------------------------------------------------------------------
# URL / Normalize Helpers
# ---------------------------------------------------------------------------


def normalize_url(url: Optional[str]) -> str:
    """Return a canonical lowercase URL with HTTPS forced and trailing slash
    removed.  Returns ``""`` for falsy input.

    Args:
        url: The raw URL string.

    Returns:
        The normalized URL string.
    """
    if not url:
        return ""
    url = url.strip().rstrip("/")
    if not url.startswith("http"):
        url = f"https://{url}"
    return url.lower()


# ---------------------------------------------------------------------------
# Snippet ID Assignment
# ---------------------------------------------------------------------------


def _assign_snippet_ids(
    snippets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign monotonically-increasing integer IDs to each snippet.

    Each snippet dict is mutated in place with an ``"id"`` key.

    Args:
        snippets: The list of snippet dicts.

    Returns:
        The same list (mutated in place) for convenience.
    """
    for idx, snip in enumerate(snippets, start=1):
        snip["id"] = idx
    return snippets


# ---------------------------------------------------------------------------
# ID Grounding
# ---------------------------------------------------------------------------


def _build_id_url_map(
    snippets: list[dict[str, Any]],
) -> dict[int, str]:
    """Build a lookup mapping snippet integer ID to its canonical URL.

    Args:
        snippets: The list of snippet dicts, each with ``id`` and ``url``.

    Returns:
        A dict of ``{id: url}``.
    """
    return {
        snip["id"]: normalize_url(snip.get("url", ""))
        for snip in snippets
        if "id" in snip
    }


def _ground_by_id(
    value: Optional[str],
    source_id: Optional[int],
    id_to_url: dict[int, str],
) -> tuple[Optional[str], Optional[str]]:
    """Ground an LLM-extracted value using its snippet source ID.

    If *value* is present but *source_id* cannot be mapped to a real URL,
    the value is discarded as a hallucination.

    Args:
        value: The field value extracted by the LLM.
        source_id: The integer snippet ID the LLM cited.
        id_to_url: Mapping from snippet ID to canonical URL.

    Returns:
        A tuple of ``(grounded_value, source_url)``.  Both are None if
        grounding fails.
    """
    if value is None:
        return None, None
    if source_id is None:
        return None, None
    if source_id not in id_to_url:
        return None, None
    url = id_to_url[source_id]
    if not url:
        return None, None
    return value, url


# ---------------------------------------------------------------------------
# Merge: XML Structured Data with LLM Data
# ---------------------------------------------------------------------------


def _merge_structured(
    enriched: EnrichedData,
    structured: dict[str, Any],
) -> EnrichedData:
    """Overwrite LLM fields with authoritative structured XML data.

    Structured values (from EDGAR / IAPD XML) are more trustworthy than
    web-scraped LLM extractions, so they take precedence for the fields
    they cover.

    Args:
        enriched: The EnrichedData from the LLM.
        structured: The dict returned by ``fetch_structured``.

    Returns:
        The same EnrichedData instance, mutated in place.
    """
    if structured.get("principal_name"):
        enriched.principal_name = structured["principal_name"]
    if structured.get("principal_title"):
        enriched.principal_title = structured["principal_title"]
    if structured.get("aum_range"):
        # EnrichedData doesn't have aum_range; it's stored separately
        pass
    if structured.get("iapd_client_count"):
        enriched.iapd_client_count = structured["iapd_client_count"]
    return enriched


def _compute_confidence(enriched: EnrichedData) -> str:
    """Compute programme confidence based on grounded high-value field count.

    3+ grounded high-value fields → High.
    1-2 grounded high-value fields → Medium.
    0 grounded high-value fields → Low.

    Args:
        enriched: The merged EnrichedData instance.

    Returns:
        One of ``"High"``, ``"Medium"``, or ``"Low"``.
    """
    count = 0
    for field in HIGH_VALUE_FIELDS:
        if getattr(enriched, field, None) is not None:
            count += 1
    if count >= 3:
        return "High"
    if count >= 1:
        return "Medium"
    return "Low"


def _has_high_value_cell(enriched: EnrichedData) -> bool:
    """Check whether at least one high-value field is populated.

    Args:
        enriched: A merged EnrichedData instance.

    Returns:
        True if any high-value field is non-None.
    """
    return any(
        getattr(enriched, field, None) is not None
        for field in HIGH_VALUE_FIELDS
    )


# ---------------------------------------------------------------------------
# Scoring & Sorting
# ---------------------------------------------------------------------------


def _score_row(row: dict[str, Any]) -> int:
    """Compute the quality score for a candidate row.

    Principal name  = 10 points.
    LinkedIn / Email = 10 points.
    Recent signal    = 10 points.
    Website          =  5 points.

    Args:
        row: The enriched row dict.

    Returns:
        The integer score.
    """
    score = 0
    if row.get("principal_name"):
        score += 10
    if row.get("principal_linkedin") or row.get("principal_email"):
        score += 10
    if row.get("recent_signal"):
        score += 10
    if row.get("website") or row.get("entity_linkedin"):
        score += 5
    return score


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    """Composite sort key: score descending, source priority, entity name.

    Args:
        row: The enriched row dict.

    Returns:
        A 3-tuple for Python's default sort ordering.
    """
    return (
        -_score_row(row),
        SOURCE_PRIORITY.get(str(row.get("discovery_source", "")), 99),
        str(row.get("entity_name", "")).lower(),
    )


# ---------------------------------------------------------------------------
# Provenance Helpers
# ---------------------------------------------------------------------------


def _get_source_type_for_field(
    field: str,
    discovery_source: str,
    structured_used: set[str],
) -> str:
    """Determine the provenance source_type for a given output field.

    Args:
        field: The output column name.
        discovery_source: The candidate's discovery source string.
        structured_used: Set of field names sourced from structured data.

    Returns:
        The source_type string for provenance logging.
    """
    if field in structured_used:
        if discovery_source == "SEC EDGAR":
            return "SEC EDGAR XML"
        if discovery_source == "IAPD":
            return "IAPD Form ADV XML"
        return "Deterministic Structured Data"
    return "Bing Web Search"


def _get_source_url_for_field(
    field: str,
    candidate_source_url: str,
    grounded_urls: dict[str, str],
    structured_used: set[str],
) -> str:
    """Determine the provenance source_url for a given output field.

    Args:
        field: The output column name.
        candidate_source_url: The candidate's discovery source URL.
        grounded_urls: Mapping from field name to grounded LLM URL.
        structured_used: Set of field names sourced from structured data.

    Returns:
        The best source URL for provenance.
    """
    if field in structured_used:
        return candidate_source_url or ""
    return grounded_urls.get(field, "")


# ---------------------------------------------------------------------------
# Address Helper
# ---------------------------------------------------------------------------


def _extract_city_state(
    address: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of (city, state) from a comma-delimited
    US address string.

    Args:
        address: A raw address string like ``"123 Main St, Austin, TX, 78701"``.

    Returns:
        A tuple of ``(city, state)``, or ``(None, None)``.
    """
    if not isinstance(address, str) or not address:
        return None, None
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) == 2 and len(parts[-1].strip()) == 2:
        return parts[0], parts[1]
    return None, None


# ---------------------------------------------------------------------------
# Query Builder
# ---------------------------------------------------------------------------


def _build_queries(
    candidate: dict[str, Any],
    city: Optional[str],
    state: Optional[str],
) -> list[str]:
    """Build an optimized list of 2 search queries per candidate.

    Queries are tailored to the discovery source to maximise signal while
    minimizing API calls.

    Args:
        candidate: The candidate dict.
        city: Extracted city from the address.
        state: Extracted state from the address.

    Returns:
        A list of query strings.
    """
    entity_name: str = candidate.get("entity_name", "")
    source: str = candidate.get("discovery_source", "")
    principal_name: Optional[str] = candidate.get("principal_name")
    is_iapd = source == "IAPD"
    is_propublica = source == "ProPublica"

    queries: list[str] = []

    if is_propublica:
        queries.append(
            f'"{entity_name}" "family office" OR "holdings" OR "capital" OR "LLC"'
        )
        if city and state:
            q2 = f'"{city}" "{state}" "family office"'
            if principal_name and isinstance(principal_name, str):
                q2 += f' "{principal_name}"'
            else:
                if entity_name.split():
                    q2 += f' "{entity_name.split()[0]}"'
            queries.append(q2)
        return queries

    if is_iapd or source == "SEC EDGAR":
        q1 = f'"{entity_name}" "CIO" OR "Managing Partner"'
        if is_iapd:
            q1 += ' OR "Number of Clients"'
        queries.append(q1)
        queries.append(
            f'"{entity_name}" "investment" OR "fund commitment" OR "linkedin.com/in"'
        )
        return queries

    return [f'"{entity_name}"']


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


def run_enrichment() -> None:
    """Execute the complete enrichment pipeline end-to-end."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    processed = _load_checkpoint()
    print(f"  Checkpoint loaded: {len(processed)} already-processed entities.")

    # ------------------------------------------------------------------
    # Load candidates
    # ------------------------------------------------------------------
    df = pd.read_csv(CANDIDATES_PATH, keep_default_na=False, na_values=[""])
    df = df.where(pd.notna(df), None)
    candidates_raw = df.to_dict("records")

    # ------------------------------------------------------------------
    # Init rejected CSV
    # ------------------------------------------------------------------
    _init_rejected_csv()

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------
    tier1_rows: list[dict[str, Any]] = []
    tier2_rows: list[dict[str, Any]] = []
    tier3_rows: list[dict[str, Any]] = []
    total = len(candidates_raw)

    for idx, candidate in enumerate(candidates_raw, start=1):
        entity_name: str = candidate.get("entity_name", "") or ""
        source: str = candidate.get("discovery_source", "")
        address: Optional[str] = candidate.get("address")
        city, state = _extract_city_state(address)

        # --- Skip already-processed ---
        if entity_name and entity_name in processed:
            print(f"[{idx}/{total}] SKIP (checkpoint): {entity_name}")
            continue

        print(f"[{idx}/{total}] Enriching: {entity_name}")

        # --------------------------------------------------------------
        # Phase 1: Pre-Filter
        # --------------------------------------------------------------
        # EDGAR real-estate drop
        if source == "SEC EDGAR" and _is_edgar_real_estate_drop(entity_name):
            _log_rejection(entity_name, source, "EDGAR Real Estate Filter")
            continue

        # IAPD Inactive drop
        entity_type_raw = str(candidate.get("entity_type", "") or "").strip()
        if entity_type_raw.upper() == "IAPD: INACTIVE":
            _log_rejection(entity_name, source, "IAPD Inactive Firm")
            continue

        # --------------------------------------------------------------
        # Phase 2: Structured Data Fetch
        # --------------------------------------------------------------
        structured = fetch_structured(candidate)

        # Mega-RIA denylist drop
        if structured.get("drop"):
            _log_rejection(
                entity_name,
                source,
                structured.get("reason", "Mega-RIA Denylist"),
            )
            continue

        # --------------------------------------------------------------
        # Phase 3: Web Search (cached Bing / DDG Lite)
        # --------------------------------------------------------------
        queries = _build_queries(candidate, city, state)
        all_snippets_raw: list[dict[str, str]] = []
        for query in queries:
            results = search_ddg(query)
            all_snippets_raw.extend(results)

        # --------------------------------------------------------------
        # Phase 4: Relevance Pre-Filter (RapidFuzz)
        # --------------------------------------------------------------
        relevant_snippets = _relevance_filter(entity_name, all_snippets_raw)
        oe_url = None
        pn_url = None
        pe_url = None
        sig_url = None
        llm_principal = None
        llm_email = None
        llm_signal = None
        llm_op_entity = None
        if not relevant_snippets:
            # No relevant snippets → skip LLM, mark as Unknown
            enriched = EnrichedData(entity_type="Unknown")
            # Still merge structured data
            enriched = _merge_structured(enriched, structured)

            # IAPD client-count check from structured
            client_count_str = structured.get("iapd_client_count", "")
            parsed_cc: Optional[int] = None
            if client_count_str:
                match = re.search(r"\d+", str(client_count_str))
                if match:
                    parsed_cc = int(match.group())

            if parsed_cc is not None and parsed_cc > MAX_CLIENT_COUNT:
                _log_rejection(
                    entity_name,
                    source,
                    f"IAPD client count > 8 ({parsed_cc})",
                )
                continue

        else:
            # ----------------------------------------------------------
            # Phase 5: ID Assignment
            # ----------------------------------------------------------
            snippets_with_ids = _assign_snippet_ids(
                list(relevant_snippets)  # copy to avoid mutating cache
            )

            # ----------------------------------------------------------
            # Phase 6: LLM Extraction
            # ----------------------------------------------------------
            enriched = extract_data_from_snippets(
                snippets_with_ids,
                entity_name=entity_name,
                principal_name=structured.get("principal_name"),
            )

            # ----------------------------------------------------------
            # Phase 7: ID Grounding & URL Resolution
            # ----------------------------------------------------------

            def _resolve_url_from_id(
                snippet_id: Optional[int],
            ) -> Optional[str]:
                """Look up a URL from a snippet's assigned integer ID."""
                if not snippet_id:
                    return None
                for s in snippets_with_ids:
                    if s.get("id") == snippet_id:
                        return s.get("url")
                return None

            def _ground_field(
                value: Optional[str],
                source_id: Optional[int],
            ) -> Optional[str]:
                """Soft-ground an LLM value: keep it even if the ID is missing.

                The value is retained for completeness; a missing source_id
                simply means no provenance URL can be mapped.
                """
                if not value:
                    return None
                return value

            llm_principal = _ground_field(
                enriched.principal_name,
                enriched.principal_name_source_id,
            )
            llm_email = _ground_field(
                enriched.principal_email,
                enriched.principal_email_source_id,
            )
            llm_signal = _ground_field(
                enriched.recent_signal,
                enriched.signal_source_id,
            )
            llm_op_entity = _ground_field(
                enriched.operating_entity_name,
                enriched.operating_entity_source_id,
            )

            if enriched.operating_entity_source_id:
                oe_url = _resolve_url_from_id(
                    enriched.operating_entity_source_id
                )
            if enriched.principal_name_source_id:
                pn_url = _resolve_url_from_id(
                    enriched.principal_name_source_id
                )
            if enriched.principal_email_source_id:
                pe_url = _resolve_url_from_id(
                    enriched.principal_email_source_id
                )
            if enriched.signal_source_id:
                sig_url = _resolve_url_from_id(
                    enriched.signal_source_id
                )

            if source == "ProPublica" and llm_op_entity and oe_url:
                base_notes = enriched.verification_notes or ""
                enriched.verification_notes = (
                    f"Operating entity for {entity_name}. {base_notes}"
                )
                entity_name = llm_op_entity

            # ----------------------------------------------------------
            # Phase 8: Merge XML Structured with LLM
            # ----------------------------------------------------------
            enriched = _merge_structured(enriched, structured)

            # Dependent-field clearing: if principal_name is None, clear
            # related fields
            if enriched.principal_name is None:
                enriched.principal_title = None
                enriched.principal_linkedin = None
                enriched.principal_email = None
                enriched.principal_email_source_id = None

            # IAPD client-count hard-drop (from structured)
            client_count_str = (
                enriched.iapd_client_count
                or structured.get("iapd_client_count", "")
            )
            parsed_cc = None
            if client_count_str:
                match = re.search(r"\d+", str(client_count_str))
                if match:
                    parsed_cc = int(match.group())

            if parsed_cc is not None and parsed_cc > MAX_CLIENT_COUNT:
                _log_rejection(
                    entity_name,
                    source,
                    f"IAPD client count > 8 ({parsed_cc})",
                )
                continue

            # Enforce casing on entity_type
            if enriched.entity_type:
                enriched.entity_type = enriched.entity_type.strip().upper()

            # Hard drops for MFO / Disqualified
            if enriched.entity_type == "MFO":
                _log_rejection(
                    entity_name, source, "LLM classified as MFO"
                )
                continue
            if enriched.entity_type == "Disqualified":
                _log_rejection(
                    entity_name, source, "LLM classified as Disqualified"
                )
                continue

        # --------------------------------------------------------------
        # Compute confidence
        # --------------------------------------------------------------
        confidence = _compute_confidence(enriched)
        enriched.confidence = confidence

        # --------------------------------------------------------------
        # Build enriched row
        # --------------------------------------------------------------
        # Track which fields came from structured vs LLM
        structured_used: set[str] = set()
        if structured.get("principal_name"):
            structured_used.add("principal_name")
        if structured.get("principal_title"):
            structured_used.add("principal_title")
        if structured.get("aum_range"):
            structured_used.add("aum_range")
        if structured.get("iapd_client_count"):
            structured_used.add("iapd_client_count")

        # Build the grounded URL map for LLM fields
        grounded_urls: dict[str, str] = {}
        if oe_url:
            grounded_urls["operating_entity_name"] = oe_url
        if pn_url:
            # Only if principal_name came from LLM, not structured
            if "principal_name" not in structured_used:
                grounded_urls["principal_name"] = pn_url
        if pe_url:
            grounded_urls["principal_email"] = pe_url
        if sig_url:
            grounded_urls["recent_signal"] = sig_url

        aum_range_val = (
            structured.get("aum_range")
            or candidate.get("aum_range")
        )

        row: dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": enriched.entity_type or "Unknown",
            "principal_name": enriched.principal_name,
            "principal_title": enriched.principal_title,
            "principal_linkedin": enriched.principal_linkedin,
            "principal_email": enriched.principal_email,
            "entity_linkedin": enriched.entity_linkedin,
            "website": enriched.website,
            "address": address,
            "aum_range": aum_range_val,
            "recent_signal": enriched.recent_signal,
            "signal_date": enriched.signal_date,
            "discovery_source": source,
            "source_url": candidate.get("source_url"),
            "verification_status": confidence,
            "verification_notes": enriched.verification_notes,
        }

        # Attach metadata for later stages
        row["_enriched"] = enriched
        row["_structured_used"] = structured_used
        row["_grounded_urls"] = grounded_urls
        row["_source_url"] = candidate.get("source_url", "")

        # --------------------------------------------------------------
        # Phase 9: 3-Tier Classification
        # --------------------------------------------------------------
        if enriched.entity_type == "SFO":
            tier1_rows.append(row)
        elif enriched.entity_type in ("Unknown", None):
            if _has_high_value_cell(enriched):
                tier2_rows.append(row)
            else:
                tier3_rows.append(row)
        else:
            # Fallback: treat as tier 3
            tier3_rows.append(row)

        # --------------------------------------------------------------
        # Checkpoint after successful enrichment
        # --------------------------------------------------------------
        _write_checkpoint(entity_name)

    # ------------------------------------------------------------------
    # Ranking & Top-50 Selection
    # ------------------------------------------------------------------
    tier1_rows.sort(key=_sort_key)
    tier2_rows.sort(key=_sort_key)
    tier3_rows.sort(key=_sort_key)

    # Tier 1 first, then Tier 2, then Tier 3 as backfill
    final_rows = tier1_rows[:50]
    if len(final_rows) < 50:
        needed = 50 - len(final_rows)
        final_rows.extend(tier2_rows[:needed])
    if len(final_rows) < 50:
        needed = 50 - len(final_rows)
        final_rows.extend(tier3_rows[:needed])

    # Label entity_type: Tier 1 → "SFO", Tier 2 & 3 → "SFO (Probable)"
    for row in final_rows:
        if row.get("entity_type") != "SFO":
            row["entity_type"] = "SFO (Probable)"

    # Replace all None with "Could not verify"
    final_none_str = "Could not verify"
    for row in final_rows:
        for key in list(row.keys()):
            val = row.get(key)
            if val is None or str(val).lower() in ("none", "null"):
                row[key] = final_none_str

    # Ensure all output columns exist
    for row in final_rows:
        for col in OUTPUT_COLUMNS:
            if col not in row:
                row[col] = final_none_str

    # ------------------------------------------------------------------
    # Persistence: family_offices.csv
    # ------------------------------------------------------------------
    output_df = pd.DataFrame(final_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # Provenance Logging
    # ------------------------------------------------------------------
    provenance_header = [
        "entity_name",
        "field_name",
        "source_type",
        "source_url",
        "verification_method",
        "confidence",
        "date",
    ]

    provenance_rows: list[list[str]] = []

    for row in final_rows:
        e_name = str(row.get("entity_name", ""))
        structured_used: set[str] = row.pop("_structured_used", set())
        grounded_urls: dict[str, str] = row.pop("_grounded_urls", {})
        cand_source_url: str = row.pop("_source_url", "")
        row.pop("_enriched", None)

        for col in OUTPUT_COLUMNS:
            cell_value = row.get(col, final_none_str)
            if cell_value == final_none_str:
                continue

            source_type = _get_source_type_for_field(
                col, str(row.get("discovery_source", "")), structured_used
            )
            source_url = _get_source_url_for_field(
                col, cand_source_url, grounded_urls, structured_used
            )
            if not source_url:
                continue

            provenance_rows.append(
                [
                    e_name,
                    col,
                    source_type,
                    source_url,
                    "LLM Extraction (llama-3.1-8b-instant)",
                    str(row.get("verification_status", "Low")),
                    today,
                ]
            )

    file_exists = os.path.isfile(PROVENANCE_PATH)
    with open(PROVENANCE_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(provenance_header)
        writer.writerows(provenance_rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sfo_count = sum(
        1 for r in final_rows if r.get("entity_type") == "SFO"
    )
    print(
        f"\nEnrichment Complete. "
        f"Total SFOs: {sfo_count}. "
        f"Written to {OUTPUT_PATH}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_enrichment()
