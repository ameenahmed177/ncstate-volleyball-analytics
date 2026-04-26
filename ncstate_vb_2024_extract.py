# ---- ncstate_vb_2024_extract.py ----
# Extract NC State WVB Game-by-Game table (2024 season) to CSV
# Works with a local HTML source OR a live URL.

import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import requests

def load_html(source: str) -> str:
    """
    If 'source' is a path to a local file, read it.
    Otherwise, treat it as a URL and GET it.
    """
    p = Path(source)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    else:
        resp = requests.get(source, timeout=30)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

def pick_game_by_game_table(html_text: str) -> pd.DataFrame:
    """
    Try pandas.read_html first; fall back to BeautifulSoup table matching.
    Select the table that has 'Opponent' and 'Result' columns.
    """
    tables = pd.read_html(html_text, header=0, flavor="lxml")
    # Heuristic: Opponent & Result must be present; sports stats columns often include Kills/Hit Pct/etc.
    def is_game_by_game(df: pd.DataFrame) -> bool:
        cols = {c.strip() for c in df.columns.astype(str)}
        needed = {"Opponent", "Result"}
        return needed.issubset(cols)
    candidates = [t for t in tables if is_game_by_game(t)]
    if candidates:
        return candidates[0]

    # Fallback: parse tables via BeautifulSoup, then feed the best one back to pandas
    soup = BeautifulSoup(html_text, "lxml")
    for tbl in soup.find_all("table"):
        headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
        if {"Opponent","Result"}.issubset({h.strip() for h in headers}):
            return pd.read_html(str(tbl), header=0)[0]

    raise RuntimeError("Could not find a Game-by-Game table (no Opponent/Result).")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for modeling.
    Handles minor header variations like Pts/PTS and ReAtt/RetAtt.
    """
    rename_map = {
        "Date":"date", "Opponent":"opponent", "Result":"result", "S":"sets",
        "Kills":"kills", "Errors":"errors", "Total Attacks":"total_attacks",
        "Hit Pct":"hitting_pct", "Assists":"assists", "Aces":"aces",
        "SErr":"serve_errors", "Digs":"digs",
        "ReAtt":"receive_attempts", "RetAtt":"receive_attempts",
        "RErr":"receive_errors",
        "Block Solos":"block_solos", "Block Assists":"block_assists",
        "BErr":"block_errors", "Pts":"points", "PTS":"points",
        "BHE":"ball_handling_errors"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}).copy()
    # Optional: parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def add_outcomes_and_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df["season"] = season
    if "result" in df.columns:
        df["win"] = df["result"].astype(str).str.contains("W", case=False, na=False).astype(int)
    return df

def extract_2024_to_csv(source_html_or_url: str, out_csv: str = "ncsu_volleyball_2024.csv") -> pd.DataFrame:
    html_text = load_html(source_html_or_url)
    gby = pick_game_by_game_table(html_text)
    gby = clean_columns(gby)
    gby = add_outcomes_and_season(gby, season=2024)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    gby.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with shape {gby.shape}")
    return gby

if __name__ == "__main__":
    # Example 1: from local HTML (what you uploaded)
    extract_2024_to_csv("ncstate_vb_2024_source.html", "ncsu_volleyball_2024.csv")

    # Example 2: from a live URL (paste the 2024 'Game by Game' URL below)
    # extract_2024_to_csv("https://stats.ncaa.org/teams/<your-2024-game-by-game-url>", "ncsu_volleyball_2024.csv")
