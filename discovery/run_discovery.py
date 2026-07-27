"""
Discovery pipeline orchestrator.

Runs all three fetchers in sequence, deduplicates the pooled
candidates using rapidfuzz fuzzy matching, applies a source-priority
hierarchy, preserves provenance for dropped duplicates, and writes the
final deduplicated ``candidates.csv``.
"""
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz

from discovery.normalize import write_provenance

# Priority: lower number = higher rank
_PRIORITY = {"SEC EDGAR": 0, "ProPublica": 1, "IAPD": 2}

_DATA_DIR = Path("data")
_CANDIDATES_CSV = _DATA_DIR / "candidates.csv"


def _ensure_data_dir() -> None:
    """Create the ``data/`` directory if it does not already exist."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_candidates() -> pd.DataFrame:
    """Load candidates CSV into a DataFrame, or return an empty frame."""
    if not _CANDIDATES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "entity_name",
                "principal_name",
                "address",
                "discovery_source",
                "source_url",
                "entity_type",
                "aum_range",
                "raw_payload",
            ]
        )
    df = pd.read_csv(_CANDIDATES_CSV, dtype=str)
    return df.where(pd.notna(df), None)


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy-deduplicate candidates using token-sort ratio on entity names.

    Two records are considered duplicates ***only*** when their
    ``entity_name`` similarity exceeds 90 %.  Principal-name similarity
    is intentionally ignored — multiple distinct foundations may share
    the same law firm, accounting firm, or registered agent as their
    care-of contact, and those are separate entities.
    """
    if df.empty:
        return df

    keep_mask = [True] * len(df)
    n = len(df)

    for i in range(n):
        if not keep_mask[i]:
            continue

        name_i = str(df.at[i, "entity_name"] or "")
        source_i = str(df.at[i, "discovery_source"] or "")
        priority_i = _PRIORITY.get(source_i, 99)

        for j in range(i + 1, n):
            if not keep_mask[j]:
                continue

            name_j = str(df.at[j, "entity_name"] or "")

            name_sim = fuzz.token_sort_ratio(name_i, name_j) if name_i and name_j else 0

            if name_sim > 90:
                source_j = str(df.at[j, "discovery_source"] or "")
                priority_j = _PRIORITY.get(source_j, 99)

                if priority_i <= priority_j:
                    _log_secondary_source(df, j, source_j)
                    keep_mask[j] = False
                else:
                    _log_secondary_source(df, i, source_i)
                    keep_mask[i] = False
                    break

    return df[keep_mask].reset_index(drop=True)


def _log_secondary_source(df: pd.DataFrame, idx: int, source_label: str) -> None:
    """Write provenance entries acknowledging a duplicate candidate from
    a secondary source before it is dropped."""
    name = df.at[idx, "entity_name"]
    url = str(df.at[idx, "source_url"] or "")

    for field in ("Entity Name", "Principal Name", "Address"):
        write_provenance(
            entity_name=name,
            field_name=field,
            source_type=source_label,
            source_url=url,
            verification_method="Duplicate Discovery (Secondary Source)",
            confidence="Medium",
        )


def run() -> None:
    """Entry point: execute all fetchers, deduplicate, and save results."""
    load_dotenv()
    _ensure_data_dir()

    # Clear previous run outputs so we start fresh
    for csv_file in [
        _CANDIDATES_CSV,
        _DATA_DIR / "provenance_log.csv",
        _DATA_DIR / "pipeline_errors.csv",
    ]:
        if csv_file.exists():
            csv_file.unlink()

    # Import fetchers here to avoid circular imports and to ensure
    # the environment is loaded before they read os.getenv.
    from discovery.edgar_fetcher import fetch_edgar_data
    from discovery.propublica_fetcher import fetch_propublica_data
    from discovery.iapd_fetcher import fetch_iapd_data

    print("=== Running SEC EDGAR fetcher ===")
    edgar_count = fetch_edgar_data()
    print(f"  EDGAR candidates: {edgar_count}")

    print("=== Running ProPublica fetcher ===")
    propublica_count = fetch_propublica_data()
    print(f"  ProPublica candidates: {propublica_count}")

    print("=== Running IAPD fetcher ===")
    iapd_count = fetch_iapd_data()
    print(f"  IAPD candidates: {iapd_count}")

    total_before = edgar_count + propublica_count + iapd_count
    print(f"\nTotal candidates before dedup: {total_before}")

    df = _load_candidates()
    if df.empty:
        print("No candidates to deduplicate.")
        return

    df_deduped = _deduplicate(df)
    df_deduped.to_csv(_CANDIDATES_CSV, index=False, encoding="utf-8")

    total_after = len(df_deduped)
    print(f"Discovery Complete. Total unique candidates: {total_after}")

    # Volume check: target range is 180-220
    if total_after < 180:
        print(f"  WARNING: Candidate count ({total_after}) is below the 180 minimum target.")
    elif total_after > 220:
        print(f"  WARNING: Candidate count ({total_after}) exceeds the 220 maximum target.")
    else:
        print(f"  Count within target range (180-220).")

    # Diversity check: no single source may exceed 50 %
    source_counts = df_deduped["discovery_source"].value_counts()
    dominant_pct = source_counts.iloc[0] / total_after * 100 if len(source_counts) > 0 else 0
    if dominant_pct > 50:
        print(
            f"  WARNING: {source_counts.index[0]} constitutes {dominant_pct:.1f}% of candidates "
            f"(max 50%). Source distribution: {dict(source_counts)}"
        )
    else:
        print(f"  Source diversity OK. Distribution: {dict(source_counts)}")


if __name__ == "__main__":
    run()
