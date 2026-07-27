"""
ProPublica Nonprofit Explorer fetcher.

Discovers up to 50 candidates by searching for "Family Foundation" on
the ProPublica Nonprofit Explorer API.  Each org's detail endpoint is
consulted for asset totals; only foundations with total assets of $50M
or more are kept.  These foundation records serve as beacons that the
Enrichment stage later resolves into the actual SFO entity.
"""
import os
import time
from typing import Optional

import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from discovery.normalize import normalize_record, save_candidate, log_error

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
API_BASE = "https://projects.propublica.org/nonprofits/api/v2"
MIN_ASSETS = 50_000_000
TARGET = 50
MAX_PAGES = 20


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def _http_get(url: str, headers: Optional[dict] = None) -> requests.Response:
    """HTTP GET with exponential-backoff retry and mandatory rate-limit."""
    if headers is None:
        headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(0.1)
    return resp


def _max_assets(org_detail: dict) -> int:
    """Return the highest total-assets figure found across the org record
    and its filing history.  Falls back to the org-level ``asset_amount``.
    """
    org = org_detail.get("organization", {})
    best = int(org.get("asset_amount") or 0)

    for filing in org_detail.get("filings_with_data", []) or []:
        for key in ("totassetsend", "totassetsendf", "totassetsboy", "totassetsboyf"):
            val = filing.get(key)
            if val is not None:
                best = max(best, int(val))

    return best


def _parse_organization(org_detail: dict) -> dict:
    """Extract candidate fields from a ProPublica organization **detail** dict.

    Returns keys: ``entity_name``, ``principal_name``, ``address``.
    """
    org = org_detail.get("organization", {})

    entity_name = org.get("name") or ""
    principal_name = org.get("careofname") or None
    if principal_name and principal_name.startswith("% "):
        principal_name = principal_name[2:]

    street = org.get("address") or ""
    city = org.get("city") or ""
    state = org.get("state") or ""
    zipcode = org.get("zipcode") or ""
    parts = [p for p in [street, city, state, zipcode] if p]
    address = ", ".join(parts) if parts else None

    return {"entity_name": entity_name, "principal_name": principal_name, "address": address}


def _org_url(ein) -> str:
    """Build the ProPublica organization page URL."""
    return f"https://projects.propublica.org/nonprofits/organizations/{ein}"


def fetch_propublica_data() -> int:
    """Discover up to 50 candidates from ProPublica Nonprofit Explorer.

    Returns:
        The number of candidates successfully written.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    candidates_found = 0
    page = 0

    while candidates_found < TARGET and page < MAX_PAGES:
        url = f"{API_BASE}/search.json?q=Family%20Foundation&page={page}"
        try:
            resp = _http_get(url, headers)
            data = resp.json()
        except Exception as e:
            log_error("ProPublica", None, f"Search failed on page {page}: {e}")
            break

        orgs = data.get("organizations", [])
        if not orgs:
            break

        for org in orgs:
            if candidates_found >= TARGET:
                break

            ein = org.get("ein", "")
            entity_label = str(ein)

            try:
                detail_url = f"{API_BASE}/organizations/{ein}.json"
                detail_resp = _http_get(detail_url, headers)
                detail = detail_resp.json()

                assets = _max_assets(detail)
                if assets < MIN_ASSETS:
                    continue

                raw_fields = _parse_organization(detail)
                if not raw_fields.get("entity_name"):
                    log_error("ProPublica", entity_label, "Missing organization name")
                    continue

                candidate = normalize_record(raw_fields, "ProPublica", _org_url(ein))
                save_candidate(candidate)
                candidates_found += 1

            except Exception as e:
                log_error("ProPublica", entity_label, str(e))
                continue

        page += 1

    return candidates_found
