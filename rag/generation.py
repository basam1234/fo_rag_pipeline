"""Grounded generation and verification layer using Groq and RapidFuzz."""

import os
import json
import pandas as pd
from groq import Groq
from rapidfuzz import fuzz, process
from dotenv import load_dotenv

# FIX: Explicitly load .env file for local testing
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "family_offices.csv")

# Load dataframe for verification (source of truth)
try:
    _df = pd.read_csv(CSV_PATH, keep_default_na=False)
except Exception:
    _df = pd.DataFrame()

_client = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        # 1. Try loading from .env (local dev)
        api_key = os.getenv("GROQ_API_KEY")
        
        # 2. Try loading from Streamlit Secrets (cloud)
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
                
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found in .env or Streamlit Secrets.")
        _client = Groq(api_key=api_key)
    return _client

def _build_context(records: list[dict]) -> str:
    """Construct token-safe context, omitting 'Could not verify' fields."""
    context = []
    for r in records:
        valid_fields = {}
        for k, v in r.items():
            if v != "Could not verify" and v != "" and pd.notna(v):
                valid_fields[k] = v
        context.append(valid_fields)
    return json.dumps(context, indent=2)

def generate_response(query: str, records: pd.DataFrame) -> dict:
    """Execute generation with fallback, then verify claims."""
    if records.empty:
        return {
            "summary": "No family offices in the dataset match your criteria.",
            "verified_claims": [],
            "unverified_claims": [],
            "model_used": "None",
            "partial_answer": False,
            "raw_records": []
        }
    
    records_list = records.to_dict("records")
    top_records = records_list[:3]  # Strict limit to save tokens
    context_str = _build_context(top_records)
    
    system_prompt = (
        "You are a financial intelligence AI. Answer the user's query using "
        "ONLY the provided JSON context of family office records. "
        "Do not use outside knowledge. If the context does not contain the "
        "answer, state that there is insufficient evidence. "
        "Output strictly valid JSON with this schema:\n"
        "{\n"
        "  \"summary\": \"string (1-2 sentence overview)\",\n"
        "  \"claims\": [\n"
        "    {\n"
        "      \"statement\": \"string (the fact asserted)\",\n"
        "      \"record_id\": \"string (must match entity_name from context)\",\n"
        "      \"field_used\": "
        "\"string (comma-separated field names from context)\",\n"
        "      \"exact_source_value\": "
        "\"string (the exact value(s) from the fields)\"\n"
        "    }\n"
        "  ],\n"
        "  \"sufficient_evidence\": boolean\n"
        "}"
    )
    
    client = _get_client()
    model_used = "None"
    response_content = None
    
    # --- Attempt 1: 70B ---
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str}"}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        response_content = response.choices[0].message.content
        model_used = "llama-3.3-70b-versatile"
    except Exception:
        # --- Fallback: 8B ---
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context_str}"}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            response_content = response.choices[0].message.content
            model_used = "llama-3.1-8b-instant (Fallback)"
        except Exception:
            # --- Total Failure: Graceful Degradation ---
            return {
                "summary": (
                    "The LLM generation service is temporarily unavailable. "
                    "Here are the raw records retrieved from our database "
                    "that match your query."
                ),
                "verified_claims": [],
                "unverified_claims": [],
                "model_used": "None (Service Unavailable)",
                "partial_answer": True,
                "raw_records": top_records
            }
            
    # --- Parse JSON safely ---
    try:
        data = json.loads(response_content)
    except json.JSONDecodeError:
        data = {
            "summary": "Error parsing LLM response.",
            "claims": [],
            "sufficient_evidence": False
        }
        
    # --- Python-Side Post-Verification ---
    verified_claims = []
    unverified_claims = []
    
    for claim in data.get("claims", []):
        record_id = claim.get("record_id", "")

        # Two-phase verification: first fuzzy-match the LLM's record_id
        # against actual entity_names in the CSV, then fuzzy-match each
        # claimed field value against the corresponding CSV cell value.
        # Thresholds: entity name match >90, field value match >85.
        match = process.extractOne(
            record_id, _df["entity_name"], scorer=fuzz.partial_ratio
        )
        if match and match[1] > 90:
            actual_entity = match[0]
            row = _df[_df["entity_name"] == actual_entity].iloc[0]
            
            fields = [f.strip() for f in claim.get("field_used", "").split(",")]
            exact_val = claim.get("exact_source_value", "")
            
            is_verified = False
            for field in fields:
                if field in row:
                    cell_val = str(row[field])
                    exact_val = str(exact_val)
                    # Fuzzy match LLM-extracted value against actual CSV cell
                    if fuzz.partial_ratio(
                        exact_val.lower(), cell_val.lower()
                    ) > 85:
                        is_verified = True
                        claim["source_url"] = row.get("source_url", "")
                        claim["actual_entity"] = actual_entity
                        break
            
            if is_verified:
                verified_claims.append(claim)
            else:
                claim["actual_entity"] = actual_entity
                unverified_claims.append(claim)
        else:
            unverified_claims.append(claim)
            
    partial_answer = (
        len(unverified_claims) > 0
        or not data.get("sufficient_evidence", True)
    )
    
    return {
        "summary": data.get("summary", "No summary provided."),
        "verified_claims": verified_claims,
        "unverified_claims": unverified_claims,
        "model_used": model_used,
        "partial_answer": partial_answer,
        "raw_records": top_records
    }