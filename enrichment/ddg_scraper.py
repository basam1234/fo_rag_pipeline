"""Dual-engine search scraper (Bing + DuckDuckGo Lite) with URL unwrapping and caching."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup, Tag

BING_ENDPOINT: str = "https://www.bing.com/search"
DDG_LITE_ENDPOINT: str = "https://lite.duckduckgo.com/lite/"

CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
CACHE_PATH: str = os.path.join(CACHE_DIR, "search_cache.json")

USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

REQUEST_TIMEOUT: int = 15
THROTTLE_SECONDS: float = 1.0
MAX_RESULTS: int = 3
SNIPPET_MAX_CHARS: int = 500  # Increased to give LLM enough context to read

_SESSION: requests.Session = requests.Session()
_SESSION.headers.update(HEADERS)

def _load_cache() -> dict[str, Any]:
    if not os.path.isfile(CACHE_PATH): return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except: return {}

def _save_cache(cache: dict[str, Any]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)

_cache_instance = _load_cache()

def _sanitize_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    if not url: return ""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        if url.startswith("//"): url = f"https:{url}"
        else: return ""
    if parsed.scheme == "http": url = url.replace("http://", "https://", 1)
    return url

def _unwrap_bing_url(raw_url: str) -> str:
    if "bing.com/ck/a" not in raw_url: return raw_url
    try:
        parsed = urlparse(raw_url)
        qs = parse_qs(parsed.query)
        u_val: str = qs.get("u", [""])[0]
        if not u_val: return raw_url
        if u_val.startswith("a1"): u_val = u_val[2:]
        padding = 4 - (len(u_val) % 4)
        if padding != 4: u_val += "=" * padding
        decoded = base64.b64decode(u_val).decode("utf-8")
        return _sanitize_url(decoded)
    except: return raw_url

def _fetch_bing(query: str) -> list[dict[str, str]]:
    try:
        resp = _SESSION.get(BING_ENDPOINT, params={"q": query, "count": 10, "setlang": "en-US"}, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code != 200: return []
            
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []

        for h2 in soup.find_all("h2"):
            a_tag = h2.find("a")
            if not a_tag or not isinstance(a_tag, Tag): continue

            raw_url: str = a_tag.get("href", "")
            url = _unwrap_bing_url(raw_url)
            if not url.startswith("http"): continue

            title = a_tag.get_text(strip=True)
            snippet = ""
            parent_li = h2.find_parent("li")
            if parent_li and isinstance(parent_li, Tag):
                caption_div = parent_li.find("div", class_="b_caption")
                if caption_div:
                    p_tag = caption_div.find("p")
                    if p_tag: snippet = p_tag.get_text(strip=True)
                if not snippet:
                    p_tag = parent_li.find("p")
                    if p_tag: snippet = p_tag.get_text(strip=True)

            results.append({"title": title[:SNIPPET_MAX_CHARS], "url": url, "snippet": snippet[:SNIPPET_MAX_CHARS]})
            if len(results) >= MAX_RESULTS: break
        return results
    except: return []

def _fetch_ddg_lite(query: str) -> list[dict[str, str]]:
    try:
        resp = _SESSION.post(DDG_LITE_ENDPOINT, data={"q": query}, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code != 200: return []
            
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[dict[str, str]] = []

        for link in soup.find_all("a", class_="result-link"):
            if not isinstance(link, Tag): continue
            title = link.get_text(strip=True)
            url = _sanitize_url(link.get("href", ""))
            if not url: continue

            snippet = ""
            parent_tr = link.find_parent("tr")
            if parent_tr and isinstance(parent_tr, Tag):
                snippet_td = parent_tr.find("td", class_="result-snippet")
                if snippet_td and isinstance(snippet_td, Tag): snippet = snippet_td.get_text(strip=True)

            results.append({"title": title[:SNIPPET_MAX_CHARS], "url": url, "snippet": snippet[:SNIPPET_MAX_CHARS]})
            if len(results) >= MAX_RESULTS: break
        return results
    except: return []

def search_ddg(query: str) -> list[dict[str, str]]:
    global _cache_instance
    if query in _cache_instance: return _cache_instance[query]
        
    results = _fetch_bing(query)
    if not results:
        print(f"  [Fallback] Bing returned 0 results. Trying DDG Lite for: {query[:50]}...")
        results = _fetch_ddg_lite(query)

    time.sleep(THROTTLE_SECONDS)
    _cache_instance[query] = results
    _save_cache(_cache_instance)
    return results