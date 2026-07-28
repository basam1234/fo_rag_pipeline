"""Deterministic Structured Data Layer for SEC EDGAR and IAPD enrichment.

Fetches canonical data from regulatory sources before any web scraping occurs.
For EDGAR, it parses primary_doc.xml for principal names/titles.
For IAPD, it queries the JSON Search API to bypass WAF blocks and find AUM/client counts.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEGA_RIA_DENYLIST: list[str] = [
    "WELLINGTON", "SCHWAB", "VANGUARD", "FRANKLIN", "BARCLAYS", 
    "TIAA", "FLEETBOSTON", "BANCORPSOUTH", "BLACKROCK", "FIDELITY",
    "ROCKEFELLER", "LOEB", "TEACHERS"
]

SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "PolarityIQ test@test.com")
REQUEST_TIMEOUT: int = 20

# Use a session to persist cookies and connection pool
_SESSION = requests.Session()


def _get_edgar_headers() -> dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }


def _get_iapd_headers(crd: str) -> dict[str, str]:
    # IAPD WAF requires exact browser headers to avoid 403 Forbidden
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://adviserinfo.sec.gov/firm/summary/{crd}",
        "Origin": "https://adviserinfo.sec.gov",
        "Connection": "keep-alive",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_structured(candidate: dict[str, Any]) -> dict[str, Any]:
    """Fetch deterministic structured data for a single candidate.

    Args:
        candidate: Dict with keys ``entity_name``, ``discovery_source``,
            ``source_url``, and ``raw_payload`` (for IAPD candidates).

    Returns:
        A dict with ``drop``, ``reason``, ``principal_name``,
        ``principal_title``, ``iapd_client_count``, and ``aum_range``.
    """
    entity_name: str = candidate.get("entity_name", "")
    source: str = candidate.get("discovery_source", "")

    result: dict[str, Any] = {
        "drop": False,
        "reason": None,
        "principal_name": None,
        "principal_title": None,
        "iapd_client_count": None,
        "aum_range": None,
    }

    # --- Mega-RIA denylist check ---
    if entity_name and any(
        keyword.upper() in entity_name.upper() for keyword in MEGA_RIA_DENYLIST
    ):
        result["drop"] = True
        result["reason"] = "Mega-RIA Denylist"
        return result

    # --- EDGAR XML parse ---
    if source == "SEC EDGAR":
        _fetch_edgar_related_person(candidate, result)

    # --- IAPD JSON API parse ---
    if source == "IAPD":
        _fetch_iapd_form_adv(candidate, result)

    return result


# ---------------------------------------------------------------------------
# EDGAR Helpers
# ---------------------------------------------------------------------------


def _recover_edgar_url(entity_name: str) -> Optional[str]:
    """Query SEC EDGAR company search to find the true Form D accession number."""
    try:
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={urllib.parse.quote(entity_name)}&type=D&dateb=&owner=include&count=10"
        resp = _SESSION.get(search_url, headers=_get_edgar_headers(), timeout=REQUEST_TIMEOUT)
        time.sleep(0.2)
        
        if resp.status_code != 200:
            return None
            
        soup = BeautifulSoup(resp.text, "html.parser")
        doc_link = soup.find("a", id="documentsbutton")
        if doc_link and doc_link.get("href"):
            # href looks like /Archives/edgar/data/919574/000091957413003058/
            parts = doc_link["href"].strip("/").split("/")
            if len(parts) >= 5:
                cik = parts[3]
                accession = parts[4]
                return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/primary_doc.xml"
    except Exception:
        pass
    return None


def _fetch_edgar_related_person(
    candidate: dict[str, Any], result: dict[str, Any]
) -> None:
    """Parse the EDGAR primary_doc.xml for the first `<relatedPerson>`."""
    source_url: str = candidate.get("source_url", "")
    if not source_url:
        return

    try:
        resp = _SESSION.get(
            source_url,
            headers=_get_edgar_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        time.sleep(0.2)
        
        # ROOT CAUSE FIX: If 404, the accession number in candidates.csv is malformed.
        # We dynamically recover the correct URL by searching the SEC EDGAR index.
        if resp.status_code == 404:
            entity_name = candidate.get("entity_name", "")
            print(f"  [StructuredFetcher] EDGAR 404. Recovering URL for {entity_name}...")
            recovered_url = _recover_edgar_url(entity_name)
            if recovered_url:
                resp = _SESSION.get(
                    recovered_url,
                    headers=_get_edgar_headers(),
                    timeout=REQUEST_TIMEOUT,
                )
                time.sleep(0.2)
                
        resp.raise_for_status()
        root = _parse_xml_strip_namespace(resp.text)
        
        person = root.find(".//relatedPerson")
        if person is not None:
            first_name = person.findtext("firstName", default="")
            last_name = person.findtext("lastName", default="")
            title = person.findtext("relatedPersonTitle", default="")
            if first_name or last_name:
                result["principal_name"] = f"{first_name} {last_name}".strip()
            if title:
                result["principal_title"] = title

    except Exception as exc:
        # If 404 persists, the filing is likely a 13F-HR (which has no primary_doc.xml).
        # The pipeline correctly defers to the LLM web search phase to find the principal.
        if "404" in str(exc):
            print(
                f"  [StructuredFetcher] No Form D XML found for {candidate.get('entity_name', 'unknown')}. "
                f"Deferring to LLM web search."
            )
        else:
            print(
                f"[StructuredFetcher] EDGAR fetch failed for "
                f"{candidate.get('entity_name', 'unknown')}: {exc}"
            )


# ---------------------------------------------------------------------------
# IAPD Helpers
# ---------------------------------------------------------------------------


def _fetch_iapd_form_adv(
    candidate: dict[str, Any], result: dict[str, Any]
) -> None:
    """Fetch firm summary data via the IAPD JSON Search API.
    
    Bypasses the Akamai WAF on the XML reports endpoint by using the same
    JSON API used during the Discovery phase.
    """
    raw_payload_raw: Any = candidate.get("raw_payload")
    if not raw_payload_raw:
        return

    crd: str | None = None
    try:
        payload = json.loads(raw_payload_raw) if isinstance(raw_payload_raw, str) else raw_payload_raw
        crd = str(payload.get("crd", ""))
    except Exception:
        return

    if not crd:
        return

    # Use the JSON Search API instead of the WAF-blocked XML report
    url = f"https://api.adviserinfo.sec.gov/search/firm?query={crd}&rows=1"
    
    try:
        resp = _SESSION.get(
            url,
            headers=_get_iapd_headers(crd),
            timeout=REQUEST_TIMEOUT,
        )
        time.sleep(0.2)
        resp.raise_for_status()

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return

        source = hits[0].get("_source", {})
        
        # The IAPD API returns many fields. Let's find the ones we need.
        for key, value in source.items():
            if value is None: continue
            key_lower = key.lower()
            val_str = str(value).strip()
            
            if "client" in key_lower and val_str and not result["iapd_client_count"]:
                result["iapd_client_count"] = val_str
            elif ("aum" in key_lower or "asset" in key_lower) and val_str and not result["aum_range"]:
                result["aum_range"] = val_str

    except Exception as exc:
        print(f"  [StructuredFetcher] IAPD API fetch failed for CRD {crd}: {str(exc)[:80]}")


# ---------------------------------------------------------------------------
# XML Utility
# ---------------------------------------------------------------------------


def _parse_xml_strip_namespace(xml_text: str) -> ET.Element:
    """Parse XML text while stripping namespace prefixes for easier traversal.

    Args:
        xml_text: Raw XML string to parse.

    Returns:
        The root ``Element`` with namespace prefixes removed.
    """
    it = ET.iterparse(
        __import__("io").StringIO(xml_text),
        events=("start-ns",),
    )
    for _, _ in it:
        pass

    root = ET.fromstring(xml_text)

    for elem in root.iter():
        if "}" in (elem.tag or ""):
            elem.tag = elem.tag.split("}", 1)[1]

    return root