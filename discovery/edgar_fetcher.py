"""
SEC EDGAR fetcher — Form D & 13F-HR.

Discovers up to 120 unique candidates by running three strict EFTS
keyword queries across Form D and 13F-HR filings, constructing archive
XML URLs directly from accession numbers (no separate submissions-index
call).  Each filing is parsed for issuer/manager name, address, and
related-person details.  Results are deduplicated by CIK and
hard-capped at 120 to maintain source diversity.
"""
import os
import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from discovery.normalize import normalize_record, save_candidate, log_error

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def _http_get(url: str, headers: Optional[dict] = None) -> requests.Response:
    """HTTP GET with exponential-backoff retry and mandatory rate-limit."""
    if headers is None:
        headers = {"User-Agent": SEC_USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(0.1)
    return resp


def _unpad_cik(raw_cik: str) -> str:
    """Strip leading zeros from a CIK."""
    return raw_cik.lstrip("0")


def _build_xml_url(adsh: str, cik: str) -> str:
    """Build the direct SEC Archive URL for a filing's primary XML document.

    Works for both Form D and 13F-HR filings.

    Args:
        adsh: The accession number, e.g. ``0001634667-15-000001``.
        cik: The unpadded CIK, e.g. ``1634667``.

    Returns:
        Full URL to the ``primary_doc.xml`` filing exhibit.
    """
    accession_clean = adsh.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik}/{accession_clean}/primary_doc.xml"
    )


def _find_by_local_name(element: ET.Element, local_name: str) -> Optional[ET.Element]:
    """Return the first descendant whose local tag name matches *local_name*.

    This is namespace-agnostic.
    """
    for elem in element.iter():
        tag = elem.tag.split("}", 1)[-1] if "}" in elem.tag else elem.tag
        if tag == local_name:
            return elem
    return None


def _text_of(parent: ET.Element, child_local: str) -> Optional[str]:
    """Return the stripped text of the first child with *child_local*."""
    child = _find_by_local_name(parent, child_local)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _parse_formd_xml(xml_text: str) -> dict:
    """Parse a Form D or 13F-HR XML document into a flat candidate dict.

    Tries Form D structure first (``primaryIssuer`` / ``relatedPerson``),
    then falls back to 13F-HR (``filingManager``).  For 13F-HR the
    principal name defaults to the filing-manager name when no individual
    officer is listed.

    Returns keys: ``entity_name``, ``principal_name``, ``address``.
    """
    root = ET.fromstring(xml_text)

    entity_name: Optional[str] = None
    principal_name: Optional[str] = None
    address: Optional[str] = None

    # --- Try Form D structure first ---
    issuer = _find_by_local_name(root, "primaryIssuer")
    if issuer is not None:
        entity_name = _text_of(issuer, "issuerName") or _text_of(issuer, "entityName")

    # --- Try 13F-HR structure if Form D didn't yield an entity name ---
    if not entity_name:
        filing_manager = _find_by_local_name(root, "filingManager")
        if filing_manager is not None:
            entity_name = _text_of(filing_manager, "name")
            # Address from 13F-HR
            addr_elem = _find_by_local_name(filing_manager, "address")
            if addr_elem is not None:
                street = _text_of(addr_elem, "street1") or ""
                city = _text_of(addr_elem, "city") or ""
                region = _text_of(addr_elem, "stateOrCountry") or ""
                zip_code = _text_of(addr_elem, "zipCode") or ""
                parts = [p for p in [street, city, region, zip_code] if p]
                address = ", ".join(parts) if parts else None

    # --- Form D: first related person ---
    if not principal_name:
        rp = _find_by_local_name(root, "relatedPerson")
        if rp is not None:
            name_elem = _find_by_local_name(rp, "relatedPersonName")
            if name_elem is not None:
                first = _text_of(name_elem, "firstName") or ""
                last = _text_of(name_elem, "lastName") or ""
                middle = _text_of(name_elem, "middleName") or ""
                principal_name = f"{first} {middle} {last}".strip().replace("  ", " ")
                if not principal_name:
                    principal_name = None

            if not address:
                addr_elem = _find_by_local_name(rp, "relatedPersonAddress")
                if addr_elem is not None:
                    street = _text_of(addr_elem, "street1") or ""
                    city = _text_of(addr_elem, "city") or ""
                    state = _text_of(addr_elem, "state") or ""
                    zip_code = _text_of(addr_elem, "zipCode") or ""
                    parts = [p for p in [street, city, state, zip_code] if p]
                    address = ", ".join(parts) if parts else None

    # --- 13F-HR: principal name fallback (use filing manager name) ---
    if not principal_name and entity_name:
        principal_name = entity_name

    return {"entity_name": entity_name, "principal_name": principal_name, "address": address}


def _fetch_edgar_search_page(query: str, start: int, headers: dict) -> list[dict]:
    """Fetch one page (up to 100 hits) from the EFTS search API."""
    url = (
        'https://efts.sec.gov/LATEST/search-index'
        f'?q={query}'
        f"&forms=D,13F-HR&start={start}&rows=100"
    )
    resp = _http_get(url, headers)
    data = resp.json()
    return data.get("hits", {}).get("hits", [])


# Strict queries targeting family-office entities only.
# Deliberately excludes broad terms ("Multi Family", "Single Family",
# "Private Wealth") to avoid contaminating results with real-estate funds.
_EDGAR_QUERIES = [
    '"Family Office"',
    '"Family Holdings"',
    '"Family Capital"',
]

_EDGAR_CAP = 120


def fetch_edgar_data() -> int:
    """Discover up to 120 unique candidates from SEC EDGAR Form D & 13F-HR.

    Runs three strict EFTS keyword queries against Form D and 13F-HR
    filings, deduplicates by CIK, and parses each unique filing's XML.
    Hard-caps at 120 saved candidates to maintain source diversity.

    Returns:
        The number of candidates successfully written.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    candidates_found = 0
    seen_ciks: set[str] = set()

    for query in _EDGAR_QUERIES:
        if candidates_found >= _EDGAR_CAP:
            break
        try:
            all_hits = _fetch_edgar_search_page(query, 0, headers)
        except Exception as e:
            log_error("SEC EDGAR", None, f"EFTS search for {query} failed: {e}")
            continue

        if not all_hits:
            continue

        for hit in all_hits:
            try:
                adsh = hit.get("_id", "").split(":")[0] if ":" in hit.get("_id", "") else ""
                src = hit.get("_source", {})
                ciks = src.get("ciks", [])
                display_names = src.get("display_names", [])

                if not adsh or not ciks:
                    continue

                cik = _unpad_cik(ciks[0])

                if cik in seen_ciks:
                    continue
                seen_ciks.add(cik)

                xml_url = _build_xml_url(adsh, cik)

                try:
                    xml_resp = _http_get(xml_url)
                    raw_fields = _parse_formd_xml(xml_resp.text)
                except Exception:
                    raw_fields = {"entity_name": None, "principal_name": None, "address": None}

                # Fall back to display name if XML parse didn't yield entity name
                if not raw_fields.get("entity_name") and display_names:
                    raw_fields["entity_name"] = display_names[0]

                candidate = normalize_record(raw_fields, "SEC EDGAR", xml_url)
                if candidate.entity_name:
                    save_candidate(candidate)
                    candidates_found += 1
                    if candidates_found >= _EDGAR_CAP:
                        break
                else:
                    log_error("SEC EDGAR", None, "Empty entity_name after parse")

            except Exception as e:
                entity_label = hit.get("_source", {}).get("display_names", [""])[0] or hit.get("_id", "")
                log_error("SEC EDGAR", str(entity_label), str(e))
                continue

    return candidates_found
