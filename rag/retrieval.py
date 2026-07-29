"""Hybrid retrieval engine: Pandas metadata filtering + NumPy semantic search."""

import os
import re
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_DIR = os.path.join(BASE_DIR, "data", "rag_index")

EMBEDDINGS_PATH = os.path.join(RAG_DIR, "embeddings.npy")
CORPUS_PATH = os.path.join(RAG_DIR, "rag_corpus.csv")

# Initialize model once (will be cached by Streamlit later)
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model

US_STATES = {
    'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS', 'CA': 'CALIFORNIA',
    'CO': 'COLORADO', 'CT': 'CONNECTICUT', 'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA',
    'HI': 'HAWAII', 'ID': 'IDAHO', 'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA',
    'KS': 'KANSAS', 'KY': 'KENTUCKY', 'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND',
    'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI', 'MO': 'MISSOURI',
    'MT': 'MONTANA', 'NE': 'NEBRASKA', 'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY',
    'NM': 'NEW MEXICO', 'NY': 'NEW YORK', 'NC': 'NORTH CAROLINA', 'ND': 'NORTH DAKOTA', 'OH': 'OHIO',
    'OK': 'OKLAHOMA', 'OR': 'OREGON', 'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA',
    'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE', 'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT',
    'VA': 'VIRGINIA', 'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN', 'WY': 'WYOMING'
}

STATE_NAMES_TO_ABBR = {v: k for k, v in US_STATES.items()}


def _parse_query_filters(query: str) -> dict:
    """Extract structured filters from natural language query."""
    filters = {}
    query_upper = query.upper()
    query_lower = query.lower()

    # 1. Geography
    for state_name, abbr in STATE_NAMES_TO_ABBR.items():
        if state_name in query_upper:
            filters["state"] = abbr
            break

    # 2. AUM (e.g., "> 500M", "over $1B", "1 billion")
    aum_match = re.search(r'(?:over|>|greater than|more than)\s*\$?(\d+(?:\.\d+)?)([MB])', query, re.IGNORECASE)
    if aum_match:
        val = float(aum_match.group(1))
        if aum_match.group(2).upper() == 'B':
            val *= 1000
        filters["aum_min"] = val
    
    # 3. Entity Type
    if "probable" in query_lower:
        filters["entity_type"] = "SFO (Probable)"
    elif "confirmed" in query_lower or "confirmed sfo" in query_lower:
        filters["entity_type"] = "SFO"
        
    return filters

def _filter_corpus(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply extracted filters to the corpus dataframe."""
    if not filters:
        return df
    
    mask = pd.Series([True] * len(df))
    
    if "state" in filters:
        mask &= (df["state"] == filters["state"])
    if "aum_min" in filters:
        mask &= (df["aum_max"] >= filters["aum_min"])
    if "entity_type" in filters:
        mask &= (df["entity_type"] == filters["entity_type"])
        
    return df[mask]

def search(query: str, top_k: int = 3) -> pd.DataFrame:
    """Execute hybrid search and return top K records."""
    # Load data
    embeddings = np.load(EMBEDDINGS_PATH)
    corpus_df = pd.read_csv(CORPUS_PATH, keep_default_na=False)
    
    # 1. Parse filters & Pre-filter
    filters = _parse_query_filters(query)
    filtered_df = _filter_corpus(corpus_df, filters)
    
    if filtered_df.empty:
        return pd.DataFrame()  # No matches
        
    # Get the indices of the filtered chunks in the original array
    vector_ids = filtered_df["vector_id"].values
    filtered_embeddings = embeddings[vector_ids]
    
    # 2. Semantic Search
    model = _get_model()
    # BGE requires query prefix
    query_text = "Represent this sentence for searching relevant passages: " + query
    
    # fastembed query_embed returns a generator, take the first element
    query_vec = np.array(list(model.query_embed([query_text])), dtype=np.float32)[0]
    
    # Dot product (cosine similarity since vectors are normalized)
    scores = np.dot(filtered_embeddings, query_vec)
    
    # Add scores to dataframe
    filtered_df = filtered_df.copy()
    filtered_df["score"] = scores
    
    # 3. Rank (Group by entity, take max score if multiple chunks exist)
    # Sort by score descending, then drop duplicates keeping the highest score
    ranked_df = filtered_df.sort_values("score", ascending=False).drop_duplicates(subset=["entity_name"])
    
    return ranked_df.head(top_k)

if __name__ == "__main__":
    # Test function
    results = search("Single family offices in Texas with over $100M AUM")
    print(results[["entity_name", "score", "chunk_type"]])