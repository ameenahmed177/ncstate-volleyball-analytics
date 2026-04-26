# Data/Scripts/vb_2025_pipeline.py
# Read:  Data/Processed/ncsu_volleyball_2025_partial_cleaned.csv
# Write: Data/Processed/ncsu_volleyball_2025_standardized.csv  (OVERWRITE)

import re
import pandas as pd
from pathlib import Path

# Cutoff: rows with date < CUTOFF are TRAIN; rows with date >= CUTOFF are in TEST window.
# IMPORTANT CHANGE: For rows >= CUTOFF we KEEP any real values present (no zero-wipe).
# Unknown future rows remain blank (NA).
CUTOFF = pd.Timestamp("2025-10-17")

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED  = SCRIPT_DIR.parent / "Processed"
IN_2025    = PROCESSED / "ncsu_volleyball_2025_cleaned.csv"
OUT_2025   = PROCESSED / "ncsu_volleyball_2025_standardized.csv"

INT_COLS = [
    "sets","kills","errors","total_attacks","assists","aces",
    "serve_errors","digs","receive_attempts","receive_errors",
    "block_solos","block_assists","block_errors","ball_handling_errors","win"
]
FLOAT_COLS = ["hitting_pct","points"]  # points is derived; see below

ORDER = ["season","date","opponent","result","win","sets","kills","errors",
         "total_attacks","hitting_pct","assists","aces","serve_errors","digs",
         "receive_attempts","receive_errors","block_solos","block_assists",
         "block_errors","points","ball_handling_errors"]

def strip_to_number(s, allow_float=True):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip()
    if s == "":
        return pd.NA
    pat = r"-?\d+(?:\.\d+)?" if allow_float else r"-?\d+"
    m = re.search(pat, s)
    return m.group(0) if m else pd.NA

def to_int_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x.map(lambda v: strip_to_number(v, allow_float=False)), errors="coerce").astype("Int64")

def to_float_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x.map(lambda v: strip_to_number(v, allow_float=True)), errors="coerce").astype("Float64")

def compute_points(df: pd.DataFrame) -> pd.Series:
    # points = kills + aces + block_solos + 0.5 * block_assists
    for c in ["kills","aces","block_solos","block_assists"]:
        if c not in df.columns:
            df[c] = pd.NA
    return (
        df["kills"].astype("Float64")
        + df["aces"].astype("Float64")
        + df["block_solos"].astype("Float64")
        + 0.5 * df["block_assists"].astype("Float64")
    )

def main():
    if not IN_2025.exists():
        raise FileNotFoundError(f"Missing input: {IN_2025}\nCWD={Path.cwd()}")

    df = pd.read_csv(IN_2025)

    # Trim text (for real strings) and preserve missing values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)

    # Normalize bogus zeros-as-text in result to NA (e.g., "0", "0.0")
    if "result" in df.columns:
        def _normalize_result(v):
            if pd.isna(v):
                return pd.NA
            if isinstance(v, (int, float)):
                return pd.NA if float(v) == 0.0 else str(v)
            s = str(v).strip()
            if re.fullmatch(r"0+(?:\.0+)?", s):
                return pd.NA
            return s if s != "" else pd.NA
        df["result"] = df["result"].map(_normalize_result)

    # Drop subtotal/header junk rows
    if "opponent" in df.columns:
        junk = df["opponent"].str.fullmatch(r"(Defensive(?: Totals)?)|(Totals)", case=False, na=False)
        df = df[~junk].copy()

    # Parse date robustly
    if "date" in df.columns:
        df["_date_text"] = df["date"].astype(str)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")  # pandas infers format
        mask_nat = df["date"].isna() & df["_date_text"].notna()
        if mask_nat.any():
            def _try_fallback(s):
                if not isinstance(s, str):
                    return pd.NaT
                # MDY with optional time
                m = re.search(r"(\d{1,2}/\d{1,2}/\d{4}(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?)", s, flags=re.IGNORECASE)
                if m:
                    return pd.to_datetime(m.group(1), errors="coerce")
                # MDY date only
                m2 = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", s)
                if m2:
                    return pd.to_datetime(m2.group(1), format="%m/%d/%Y", errors="coerce")
                # ISO
                m3 = re.search(r"(\d{4}-\d{2}-\d{2})", s)
                if m3:
                    return pd.to_datetime(m3.group(1), format="%Y-%m-%d", errors="coerce")
                return pd.NaT
            df.loc[mask_nat, "date"] = df.loc[mask_nat, "_date_text"].map(_try_fallback)
        df = df.drop(columns=["_date_text"])

    # Ensure season
    if "season" not in df.columns:
        df["season"] = 2025
    else:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(2025).astype("Int64")

    # If SMU missing date, set scheduled date
    if {"opponent","date"}.issubset(df.columns):
        smu_mask = df["opponent"].str.fullmatch(r"SMU", case=False, na=False) & df["date"].isna()
        df.loc[smu_mask, "date"] = pd.Timestamp("2025-11-29")

    # Split by cutoff:
    #   TRAIN:  date <  CUTOFF
    #   TEST:   date >= CUTOFF (played or scheduled)
    #   FUTURE: date is NA (unknown scheduling)
    is_train  = df["date"].notna() & (df["date"] <  CUTOFF)
    is_test   = df["date"].notna() & (df["date"] >= CUTOFF)
    is_future = df["date"].isna()

    train  = df.loc[is_train].copy()
    test   = df.loc[is_test].copy()
    future = df.loc[is_future].copy()

    # ---- TRAIN cleaning ----
    for c in INT_COLS:
        if c in train.columns:
            train[c] = to_int_series(train[c]).fillna(0)
    for c in FLOAT_COLS:
        if c in train.columns:
            train[c] = to_float_series(train[c]).fillna(0.0)

    # Recompute win from result for TRAIN (only where result is present)
    if "result" in train.columns:
        train["win"] = pd.NA
        mask_train_played = train["result"].notna() & train["result"].astype(str).str.strip().ne("")
        if mask_train_played.any():
            train.loc[mask_train_played, "win"] = (
                train.loc[mask_train_played, "result"].astype(str)
                .str.contains(r"\bW\b", case=False, na=False)
                .astype("Int64")
            )
        train["win"] = train["win"].astype("Int64")

    # Compute points for TRAIN
    train["points"] = compute_points(train).fillna(0.0)

    # Clamp obviously-bad block_errors
    if "block_errors" in train.columns:
        be = pd.to_numeric(train["block_errors"], errors="coerce")
        train.loc[be > 20, "block_errors"] = 0
        train["block_errors"] = train["block_errors"].astype("Int64")

    # ---- TEST cleaning (coerce types; do NOT zero-fill unknowns) ----
    if not test.empty:
        for c in INT_COLS:
            if c in test.columns:
                test[c] = pd.to_numeric(test[c].map(lambda x: strip_to_number(x, allow_float=False)), errors="coerce").astype("Int64")
        for c in FLOAT_COLS:
            if c in test.columns:
                test[c] = pd.to_numeric(test[c].map(lambda x: strip_to_number(x, allow_float=True)), errors="coerce")

        # Played vs scheduled in TEST (played = has a non-empty result)
        has_result = ("result" in test.columns)
        played_test_mask = (test["result"].notna() & test["result"].astype(str).str.strip().ne("")) if has_result else pd.Series(False, index=test.index)

        # Recompute win for played TEST matches; leave NA for scheduled
        if has_result:
            test["win"] = pd.NA
            if played_test_mask.any():
                test.loc[played_test_mask, "win"] = (
                    test.loc[played_test_mask, "result"].astype(str)
                    .str.contains(r"\bW\b", case=False, na=False)
                    .astype("Int64")
                )
            test["win"] = test["win"].astype("Int64")

        # Recompute points for TEST when stats are present (NA if components are NA)
        for req in ["kills","aces","block_solos","block_assists"]:
            if req not in test.columns:
                test[req] = pd.NA
        test["points"] = (
            pd.to_numeric(test["kills"], errors="coerce").astype(float)
            + pd.to_numeric(test["aces"], errors="coerce").astype(float)
            + pd.to_numeric(test["block_solos"], errors="coerce").astype(float)
            + 0.5 * pd.to_numeric(test["block_assists"], errors="coerce").astype(float)
        )

        # Clamp block_errors in TEST; 0 only for played matches, else leave NA
        if "block_errors" in test.columns:
            be_t = pd.to_numeric(test["block_errors"], errors="coerce")
            test.loc[be_t > 20, "block_errors"] = pd.NA
            if played_test_mask.any():
                test.loc[played_test_mask, "block_errors"] = (
                    pd.to_numeric(test.loc[played_test_mask, "block_errors"], errors="coerce")
                    .fillna(0).astype("Int64")
                )
            if (~played_test_mask).any():
                test.loc[~played_test_mask, "block_errors"] = (
                    test.loc[~played_test_mask, "block_errors"]
                    .where(pd.notna(test.loc[~played_test_mask, "block_errors"]), pd.NA)
                    .astype("Int64")
                )

        # For TEST: if played, fill missing ball_handling_errors with 0; scheduled stays NA
        if "ball_handling_errors" in test.columns:
            test["ball_handling_errors"] = pd.to_numeric(test["ball_handling_errors"], errors="coerce")
            if played_test_mask.any():
                test.loc[played_test_mask, "ball_handling_errors"] = (
                    test.loc[played_test_mask, "ball_handling_errors"].fillna(0).astype("Int64")
                )
            if (~played_test_mask).any():
                test.loc[~played_test_mask, "ball_handling_errors"] = (
                    test.loc[~played_test_mask, "ball_handling_errors"]
                    .where(pd.notna(test.loc[~played_test_mask, "ball_handling_errors"]), pd.NA)
                    .astype("Int64")
                )

    # ---- FUTURE: leave “untouched” (only keep season/date/opponent) ----
    keep_cols = {"season","date","opponent"}
    for c in df.columns:
        if c not in keep_cols:
            future[c] = pd.NA
    for c in INT_COLS:
        if c in future.columns:
            future[c] = future[c].astype("Int64")
    for c in FLOAT_COLS:
        if c in future.columns:
            future[c] = future[c].astype("Float64")

    # Merge back, order, write
    out = pd.concat([train, test, future], ignore_index=True).sort_values(["date","opponent"], na_position="last")

    # Drop trailing junk rows with no date and no opponent (or purely numeric opponent)
    if {"date","opponent"}.issubset(out.columns):
        mask_junk = out["date"].isna() & (
            out["opponent"].isna()
            | out["opponent"].astype(str).str.strip().eq("")
            | out["opponent"].astype(str).str.fullmatch(r"\d+")
        )
        if mask_junk.any():
            out = out.loc[~mask_junk].copy()

    for col in ORDER:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[ORDER]

    # Ensure date-only in output CSV
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date

    OUT_2025.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_2025, index=False)

    print(f"Overwrote: {OUT_2025}  shape={out.shape}")
    # Summary: TEST is any row with a known date >= CUTOFF (played or scheduled)
    print(f"Train rows (< {CUTOFF.date()}): {train.shape[0]}  |  Test rows (>= {CUTOFF.date()}): {test.shape[0]}  |  Future/unknown: {future.shape[0]}")

if __name__ == "__main__":
    main()
