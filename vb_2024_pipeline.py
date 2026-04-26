# Data/Scripts/vb_2024_pipeline.py
import re, time
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED  = SCRIPT_DIR.parent / "Processed"

# Read FROM the standardized file if you want to keep re-running in-place,
# or switch IN_2024 to the extractor output if you prefer that as the source.
IN_2024  = PROCESSED / "ncsu_volleyball_2024_standardized.csv"   # source for in-place updates
OUT_2024 = IN_2024                                               # overwrite in place

INT_COLS = [
    "sets","kills","errors","total_attacks","assists","aces",
    "serve_errors","digs","receive_attempts","receive_errors",
    "block_solos","block_assists","block_errors","ball_handling_errors","win"
]
FLOAT_COLS = ["hitting_pct","points"]
ORDER = ["season","date","opponent","result","win","sets","kills","errors",
         "total_attacks","hitting_pct","assists","aces","serve_errors","digs",
         "receive_attempts","receive_errors","block_solos","block_assists",
         "block_errors","points","ball_handling_errors"]

def strip_to_number(s, allow_float=True):
    if pd.isna(s): return pd.NA
    s = str(s)
    m = re.search(r"-?\d+(\.\d+)?", s) if allow_float else re.search(r"-?\d+", s)
    return m.group(0) if m else pd.NA

def main():
    if not IN_2024.exists():
        raise FileNotFoundError(f"Missing input: {IN_2024}\nCWD: {Path.cwd()}")

    before_mtime = IN_2024.stat().st_mtime

    df = pd.read_csv(IN_2024)

    # Trim text columns (preserve NA; do not coerce to "nan")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)

    # Drop NCAA subtotal/header junk
    if "opponent" in df.columns:
        junk = df["opponent"].astype(str).str.fullmatch(r"(Defensive(?: Totals)?)|(Totals)", case=False, na=False)
        df = df[~junk].copy()

    # Parse date (keep as datetime64; downstream code expects datetimes)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Coerce numerics + fill blanks with 0
    for c in INT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda x: strip_to_number(x, allow_float=False))
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda x: strip_to_number(x, allow_float=True))
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Recompute POINTS = kills + aces + block_solos + 0.5*block_assists
    for req in ["kills","aces","block_solos","block_assists"]:
        if req not in df.columns:
            df[req] = 0
    df["points"] = (
        df["kills"].astype(float)
        + df["aces"].astype(float)
        + df["block_solos"].astype(float)
        + 0.5 * df["block_assists"].astype(float)
    )

    # Ensure season/win
    if "season" not in df.columns:
        df["season"] = 2024
    if "win" not in df.columns and "result" in df.columns:
        df["win"] = df["result"].astype(str).str.contains("W", case=False, na=False).astype("Int64")

    # Ensure required cols + order
    for col in ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[ORDER]

    # IMPORTANT: make sure Excel is CLOSED or Windows will block the write!
    OUT_2024.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_2024, index=False)

    # Confirm overwrite
    time.sleep(0.2)  # tiny delay for filesystem timestamp
    after_mtime = OUT_2024.stat().st_mtime
    print(f"Overwrote: {OUT_2024}")
    print(f"Modified time changed? {after_mtime > before_mtime}")

if __name__ == "__main__":
    main()
