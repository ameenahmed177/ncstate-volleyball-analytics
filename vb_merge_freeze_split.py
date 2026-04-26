# Data/Scripts/vb_merge_freeze_split.py
import pandas as pd
from pathlib import Path

# Freeze cutoff for train/test split (inclusive for train, exclusive for test)
FREEZE = pd.Timestamp("2025-10-11")

# Build paths relative to this script so it runs from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent          # ../Data
PROCESSED = DATA_DIR / "Processed"

P24 = PROCESSED / "ncsu_volleyball_2024_standardized.csv"
P25 = PROCESSED / "ncsu_volleyball_2025_standardized.csv"

OUT_COMBINED = PROCESSED / "ncsu_volleyball_combined.csv"
OUT_TRAIN    = PROCESSED / "ncsu_volleyball_train_through_2025-10-11.csv"
OUT_TEST     = PROCESSED / "ncsu_volleyball_holdout_after_2025-10-11.csv"

# Required schema (kept in this order)
REQUIRED = [
    "season","date","opponent","result","win","sets","kills","errors",
    "total_attacks","hitting_pct","assists","aces","serve_errors","digs",
    "receive_attempts","receive_errors","block_solos","block_assists",
    "block_errors","points","ball_handling_errors"
]

def load_csv(path: Path, season_hint: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}\nCWD: {Path.cwd()}")
    df = pd.read_csv(path)

    # Parse date (coerce bad values) and normalize to date-only (not datetime with time)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Normalize season: coerce to numeric; if missing/invalid and date exists, use date.year
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        if "date" in df.columns:
            # Use year from date when season is NA or clearly invalid
            mask_fix = pd.notna(df["date"]) & (df["season"].isna() | (df["season"] < 2000))
            if mask_fix.any():
                df.loc[mask_fix, "season"] = pd.to_datetime(df.loc[mask_fix, "date"]).dt.year
        df["season"] = df["season"].fillna(season_hint).astype(int)
    else:
        df["season"] = season_hint

    # If win not present, derive from result (look for a standalone 'W')
    if "win" not in df.columns and "result" in df.columns:
        df["win"] = df["result"].astype(str).str.contains(r"\bW\b", case=False, na=False).astype("Int64")

    # Ensure required columns exist
    for c in REQUIRED:
        if c not in df.columns:
            df[c] = pd.NA

    # Reorder uniformly
    df = df[REQUIRED]
    return df

def main():
    v24 = load_csv(P24, 2024)
    v25 = load_csv(P25, 2025)

    # Combine, sort, and dedupe by (season, date, opponent)
    combined = pd.concat([v24, v25], ignore_index=True)
    # Convert date back to datetime for sorting (keeps CSVs as date later)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined
        .sort_values(["season", "date", "opponent"], na_position="last")
        .drop_duplicates(subset=["season", "date", "opponent"], keep="first")
    )

    # Write combined (with date-only)
    combined_out = combined.copy()
    combined_out["date"] = combined_out["date"].dt.date
    PROCESSED.mkdir(parents=True, exist_ok=True)
    combined_out.to_csv(OUT_COMBINED, index=False)

    # Time-based split (≤ FREEZE -> train, > FREEZE -> test)
    train = combined[combined["date"] <= pd.Timestamp(FREEZE)].copy()
    test  = combined[combined["date"] >  pd.Timestamp(FREEZE)].copy()
    # Write splits with date-only
    for df, path in [(train, OUT_TRAIN), (test, OUT_TEST)]:
        d = df.copy()
        d["date"] = d["date"].dt.date
        d.to_csv(path, index=False)

    # Helpful console summary
    print(f"Combined saved: {OUT_COMBINED}  shape={combined_out.shape}")
    print(f"Train (≤ {FREEZE.date()}): {train.shape}  dates: {train['date'].min()} → {train['date'].max()}")
    print(f"Test  (> {FREEZE.date()}): {test.shape}   dates: {test['date'].min()} → {test['date'].max()}")

    if "win" in train.columns:
        print("\nClass balance:")
        print("  Train win counts:\n", train["win"].value_counts(dropna=False))
        print("  Test  win counts:\n",  test["win"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
