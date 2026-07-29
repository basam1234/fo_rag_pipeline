"""Builds the RAG index from family_offices.csv using fastembed (ONNX)."""

import os
import re
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "family_offices.csv")
RAG_DIR = os.path.join(DATA_DIR, "rag_index")

EMBEDDINGS_PATH = os.path.join(RAG_DIR, "embeddings.npy")
CORPUS_PATH = os.path.join(RAG_DIR, "rag_corpus.csv")

MODEL_NAME = "BAAI/bge-small-en-v1.5"

def _parse_aum(aum_str: str) -> tuple[int, int]:
    """Parse AUM string into min/max numeric values (in millions)."""
    if not aum_str or aum_str == "Could not verify":
        return 0, 99999

    matches = re.findall(
        r"\$?(\d+(?:\.\d+)?)([MB])", aum_str, re.IGNORECASE
    )
    if not matches:
        return 0, 99999
    
    vals = []
    for val, unit in matches:
        num = float(val)
        if unit.upper() == 'B':
            num *= 1000
        vals.append(num)
    
    return int(min(vals)), int(max(vals))

def _parse_state(address: str) -> str:
    """Extract 2-letter state code from address string."""
    if not address or address == "Could not verify":
        return "UNKNOWN"
    match = re.search(r',\s*([A-Z]{2})\s\d{5}', address)
    return match.group(1) if match else "UNKNOWN"

def _build_chunks(row: pd.Series) -> list[dict]:
    """Construct Profile and optionally Signal chunks for a record."""

    def _mask(val):
        return "[N/A]" if val == "Could not verify" else val
    
    profile_text = (
        f"Entity: {_mask(row['entity_name'])}. "
        f"Type: {_mask(row['entity_type'])}. "
        f"Location: {_mask(row['state'])}. "
        f"Principal: {_mask(row['principal_name'])} ({_mask(row['principal_title'])}). "
        f"AUM: {_mask(row['aum_range'])}. "
        f"Website: {_mask(row['website'])}. "
        f"Recent Signal: {_mask(row['recent_signal'])}."
    )
    
    chunks = [{
        "entity_name": row["entity_name"],
        "chunk_type": "profile",
        "text": profile_text
    }]
    
    if row["recent_signal"] != "Could not verify":
        signal_text = (
            f"Recent activity for {row['entity_name']}: "
            f"{row['recent_signal']} ({row['signal_date']})."
        )
        chunks.append({
            "entity_name": row["entity_name"],
            "chunk_type": "signal",
            "text": signal_text
        })
        
    return chunks

def run() -> None:
    """Execute the indexing pipeline."""
    os.makedirs(RAG_DIR, exist_ok=True)
    
    # CRITICAL: keep_default_na=False ensures "Could not verify" stays a string
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    
    # Extract Metadata
    df["state"] = df["address"].apply(_parse_state)
    aum_bounds = df["aum_range"].apply(_parse_aum)
    df["aum_min"] = [x[0] for x in aum_bounds]
    df["aum_max"] = [x[1] for x in aum_bounds]
    
    # Build Chunks
    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(_build_chunks(row))
        
    chunks_df = pd.DataFrame(all_chunks)
    
    # Generate Embeddings
    print(f"Loading ONNX model: {MODEL_NAME}...")
    model = TextEmbedding(model_name=MODEL_NAME)
    
    print(f"Embedding {len(chunks_df)} chunks...")
    # fastembed returns a generator, must consume it into a list first
    embeddings_list = list(model.embed(chunks_df["text"].tolist()))
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    # L2 Normalize for dot product cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Persist
    np.save(EMBEDDINGS_PATH, embeddings)
    
    # Add vector_id to corpus for reference
    chunks_df["vector_id"] = range(len(chunks_df))
    
    # Merge original df metadata into chunks_df for retrieval filtering
    # We'll merge on entity_name to get state/aum bounds into the chunk metadata
    merged_df = pd.merge(
        chunks_df,
        df[["entity_name", "state", "aum_min", "aum_max", "entity_type"]],
        on="entity_name",
    )
    
    merged_df.to_csv(CORPUS_PATH, index=False)
    print(f"Index built successfully. Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    run()