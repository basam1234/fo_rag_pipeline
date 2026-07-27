"""
SEC IAPD (Investment Adviser Public Disclosure) fetcher.

Discovers up to 100 firms by running five focused keyword queries
against the IAPD firm-search API.  Each hit's ``firm_ia_address_details``
JSON is parsed for address, and results are deduplicated by
``firm_source_id`` across queries.  Hard-capped at 100.
"""
import json
import os
import time
from typing import Optional

import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from discovery.normalize import normalize_record, save_candidate, log_error

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
_QUERIES = [
    '"Family Office"',
    '"Family Wealth"',
    '"Family Investment"',
    '"Family Capital"',
    '"Family Trust"',
]
_IAPD_CAP = 100


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def _http_get(url: str, headers: Optional[dict] = None) -> requests.Response:
    """HTTP GET with exponential-backoff retry and mandatory rate-limit."""
    if headers is None:
        headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept": "application/json",
        }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(0.1)
    return resp


def _parse_address(addr_json: Optional[str]) -> Optional[str]:
    """Parse the IAPD ``firm_ia_address_details`` JSON string into a
    human-readable address line."""
    if not addr_json:
        return None
    try:
        addr_obj = json.loads(addr_json) if isinstance(addr_json, str) else addr_json
    except (json.JSONDecodeError, TypeError):
        return None

    office = addr_obj.get("officeAddress", {})
    parts = [
        office.get("street1", ""),
        office.get("street2", ""),
        office.get("city", ""),
        office.get("state", ""),
        office.get("postalCode", ""),
        office.get("country", ""),
    ]
    combined = ", ".join(p for p in parts if p)
    return combined if combined else None


def _parse_hit(hit: dict) -> dict:
    """Extract candidate fields from a single IAPD search hit."""
    source = hit.get("_source", {})

    entity_name = source.get("firm_name")
    crd = source.get("firm_source_id")
    address = _parse_address(source.get("firm_ia_address_details"))

    entity_type: Optional[str] = None
    scope = source.get("firm_ia_scope", "")
    if scope:
        entity_type = f"IAPD: {scope}"

    return {
        "entity_name": entity_name,
        "address": address,
        "entity_type": entity_type,
        "crd": crd,
    }


def fetch_iapd_data() -> int:
    """Discover up to 100 firms from the IAPD firm-search API.

    Runs five focused keyword queries, deduplicating by firm_source_id
    across queries.  Hard-caps at 100 saved candidates.

    Returns:
        The number of candidates successfully written.
    """
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json",
    }
    candidates_found = 0
    seen_crds: set[str] = set()

    for query_term in _QUERIES:
        if candidates_found >= _IAPD_CAP:
            break

        query_url = (
            "https://api.adviserinfo.sec.gov/search/firm"
            f"?query={query_term}&rows=100"
        )

        try:
            resp = _http_get(query_url, headers)
            data = resp.json()
        except Exception as e:
            log_error("IAPD", None, f"IAPD search failed for {query_term}: {e}")
            continue

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            continue

        for hit in hits:
            if candidates_found >= _IAPD_CAP:
                break

            try:
                raw_fields = _parse_hit(hit)

                entity_name = raw_fields.get("entity_name")
                crd_value = raw_fields.get("crd", "")

                if crd_value and crd_value in seen_crds:
                    continue
                if crd_value:
                    seen_crds.add(crd_value)

                source_url = (
                    f"https://adviserinfo.sec.gov/firm/summary/{crd_value}"
                    if crd_value else ""
                )

                if not entity_name:
                    log_error("IAPD", str(crd_value), "Missing entity name")
                    continue

                candidate = normalize_record(raw_fields, "IAPD", source_url)
                save_candidate(candidate)
                candidates_found += 1

            except Exception as e:
                crd_label = hit.get("_source", {}).get("firm_source_id", "") or hit.get("_id", "")
                log_error("IAPD", str(crd_label), str(e))
                continue

    return candidates_found
