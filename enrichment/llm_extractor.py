"""Groq LLM extraction logic using raw SDK, anchors, and ID-based grounding."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

class EnrichedData(BaseModel):
    operating_entity_name: Optional[str] = None
    operating_entity_source_id: Optional[int] = None
    entity_type: Optional[str] = None
    principal_name: Optional[str] = None
    principal_name_source_id: Optional[int] = None
    principal_title: Optional[str] = None
    principal_linkedin: Optional[str] = None
    principal_email: Optional[str] = None
    principal_email_source_id: Optional[int] = None
    entity_linkedin: Optional[str] = None
    website: Optional[str] = None
    recent_signal: Optional[str] = None
    signal_date: Optional[str] = None
    signal_source_id: Optional[int] = None
    iapd_client_count: Optional[str] = None
    verification_notes: Optional[str] = None
    confidence: Optional[str] = None

_client_instance: Optional[Groq] = None

def _get_client() -> Groq:
    global _client_instance
    if _client_instance is None:
        api_key: str = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _client_instance = Groq(api_key=api_key)
    return _client_instance

def extract_data_from_snippets(snippets: list[dict[str, Any]], entity_name: str = "", principal_name: Optional[str] = None) -> EnrichedData:
    try:
        client = _get_client()
        
        anchor = f"Target entity: '{entity_name}'."
        if principal_name:
            anchor += f" Known verified principal: '{principal_name}'."
            
        system_prompt = (
            f"You are a financial data extraction AI. {anchor} "
            "Only extract facts if the snippet text explicitly names this exact entity. "
            "Do NOT extract historical figures or fund names. "
            "For provenance, output the integer id of the snippet that provided each piece of information. "
            "If a fact is not supported by any snippet, use null for both the value and its source_id. "
            "If >8 clients or multiple families, entity_type='MFO'. "
            "If disqualified or not a family office, entity_type='Disqualified'. "
            "Respond ONLY with valid JSON matching this schema: "
            '{"operating_entity_name": str|null, "operating_entity_source_id": int|null, "entity_type": "SFO"|"MFO"|"Unknown"|"Disqualified", "principal_name": str|null, "principal_name_source_id": int|null, "principal_title": str|null, "principal_linkedin": str|null, "principal_email": str|null, "principal_email_source_id": int|null, "entity_linkedin": str|null, "website": str|null, "recent_signal": str|null, "signal_date": str|null, "signal_source_id": int|null, "iapd_client_count": str|null, "verification_notes": str, "confidence": "High"|"Medium"|"Low"}'
        )
        
        snippets_json = json.dumps(snippets, indent=2, default=str)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Snippets:\n{snippets_json}"},
            ],
        )

        time.sleep(8.0)  # 8-second pause to respect Groq's 6,000 TPM limit

        raw_content = response.choices[0].message.content
        if not raw_content:
            return EnrichedData(entity_type="Unknown")

        data = json.loads(raw_content)
        return EnrichedData(**data)

    except Exception as exc:
        print(f"[LLM Error] {type(exc).__name__}: {str(exc)[:200]}")
        return EnrichedData(entity_type="Unknown")