#!/usr/bin/env python3
"""Extract NC State WVB Game-by-Game table (2025 season) to CSV.
Robust: prefer visible text, fallback to data-sort attribute and stripped strings.
Clears 'date' for Defensive Totals rows to avoid duplicate-date summary rows.
"""
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import requests
import re


def load_html(source: str) -> str:
    p = Path(source)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    resp = requests.get(source, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def pick_game_by_game_table(html_text: str) -> pd.DataFrame:
    soup = BeautifulSoup(html_text, "lxml")

    def table_to_df(tbl) -> pd.DataFrame:
        # Build header list
        headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
        if not headers:
            first_row = tbl.find("tr")
            if first_row:
                headers = [c.get_text(strip=True) for c in first_row.find_all(["th", "td"]) ]
                headers = [h for h in headers if h]
        if not headers:
            return None

        rows = []
        for tr in tbl.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            def _cell_text(cell):
                try:
                    txt = cell.get_text(strip=True)
                except Exception:
                    txt = ""
                if txt:
                    return txt
                if cell.has_attr('data-sort'):
                    ds = cell.get('data-sort')
                    if ds is not None:
                        return str(ds).strip()
                parts = [s for s in cell.stripped_strings]
                if parts:
                    return " ".join(parts)
                return ""

            texts = [_cell_text(c) for c in cells]
            # skip header row that repeated
            if texts == headers:
                continue
            if len(texts) < len(headers):
                texts += [""] * (len(headers) - len(texts))
            rows.append(dict(zip(headers, texts[:len(headers)])))

        if not rows:
            return None
        return pd.DataFrame(rows)

    # Helper to determine if a table looks like game-by-game
    def _is_gbg_columns(cols_set: set) -> bool:
        cols = {c.strip().lower() for c in cols_set}
        opp_candidates = {"opponent", "opp", "opponents", "team"}
        result_candidates = {"result", "res", "score", "final", "w-l", "w/l", "w-l"}
        has_opp = any(o in cols for o in opp_candidates)
        has_result = any(r in cols for r in result_candidates)
        return has_opp and has_result

    # Try BeautifulSoup tables first
    for tbl in soup.find_all("table"):
        df_tbl = table_to_df(tbl)
        if df_tbl is None:
            continue
        if _is_gbg_columns(df_tbl.columns):
            return df_tbl

    # Fallback to pandas.read_html if available
    try:
        tables = pd.read_html(html_text, header=0)
        def looks_like_gbg(df: pd.DataFrame) -> bool:
            cols = {c.strip() for c in df.columns.astype(str)}
            return _is_gbg_columns(cols)
        candidates = [t for t in tables if looks_like_gbg(t)]
        if candidates:
            return candidates[0]
    except Exception:
        pass

    # Save debug HTML locally for inspection to help diagnose missing table
    try:
        debug_path = Path(__file__).resolve().parent / "_debug_ncstate_vb_2025_source.html"
        debug_path.write_text(html_text, encoding="utf-8", errors="ignore")
        raise RuntimeError(f"Game-by-Game table not found (no Opponent/Result). Debug HTML saved to: {debug_path}")
    except Exception:
        raise RuntimeError("Game-by-Game table not found (no Opponent/Result).")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Date":"date", "Opponent":"opponent", "Result":"result", "S":"sets",
        "Kills":"kills", "Errors":"errors", "Total Attacks":"total_attacks",
        "Hit Pct":"hitting_pct", "Assists":"assists", "Aces":"aces", "SErr":"serve_errors",
        "Digs":"digs", "ReAtt":"receive_attempts", "RetAtt":"receive_attempts", "RErr":"receive_errors",
        "Block Solos":"block_solos", "Block Assists":"block_assists", "BErr":"block_errors",
        "Pts":"points", "PTS":"points", "BHE":"ball_handling_errors"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}).copy()
    # preserve raw date text for robust parsing downstream
    if "date" in df.columns:
        # Only strip actual strings; preserve missing values (do not convert NaN to 'nan')
        df["date"] = df["date"].map(lambda v: v.strip() if isinstance(v, str) else v)
    # Trim text fields
    for c in df.columns:
        if df[c].dtype == object:
            # Only strip strings so that NaN stays NaN
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def add_metadata(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df["season"] = season
    # Create nullable integer win column: True->1, False->0 for played matches; NA for missing result
    if "result" in df.columns:
        df["win"] = pd.NA
        mask = df["result"].notna()
        if mask.any():
            df.loc[mask, "win"] = (
                df.loc[mask, "result"].astype(str).str.contains(r"\bW\b", case=False, na=False).astype("Int64")
            )
        # ensure Int64 dtype
        df["win"] = df["win"].astype("Int64")
    return df


def extract_2025_to_csv(source_html_or_url: str, out_csv: str = "../Processed/ncsu_volleyball_2025.csv") -> pd.DataFrame:
    html_text = load_html(source_html_or_url)
    gbg = pick_game_by_game_table(html_text)
    gbg = clean_columns(gbg)

    # Normalize opponent column name
    if 'opponent' not in gbg.columns and 'Opponent' in gbg.columns:
        gbg = gbg.rename(columns={'Opponent':'opponent'})

    # Clear date on defensive totals rows so they don't duplicate dates
    if 'opponent' in gbg.columns and 'date' in gbg.columns:
        mask_def = gbg['opponent'].astype(str).str.fullmatch(r"(Defensive(?: Totals)?)|(Totals)", case=False, na=False)
        gbg.loc[mask_def, 'date'] = ''

    gbg = add_metadata(gbg, season=2025)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    gbg.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} with shape {gbg.shape}")
    return gbg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NC State 2025 game-by-game table to CSV from a local HTML file or URL.")
    parser.add_argument("--source", "-s", default="https://stats.ncaa.org/teams/<your-2025-game-by-game-url>", help="Local HTML file path or URL to scrape. Default is a placeholder NCAA URL — replace with the real one or pass --source.")
    parser.add_argument("--out", "-o", default="../Processed/ncsu_volleyball_2025.csv", help="Output CSV path (default: Data/Processed/ncsu_volleyball_2025.csv)")
    args = parser.parse_args()
    src = args.source
    out = args.out
    placeholder = "https://stats.ncaa.org/teams/<your-2025-game-by-game-url>"
    if src == placeholder:
        filename = "ncstate_vb_2025_source.html"
        script_dir = Path(__file__).resolve().parent
        candidates = [
            Path(filename),
            script_dir / filename,
            script_dir.parent / "Raw" / filename,
            script_dir.parent / filename,
            Path("../Raw") / filename,
        ]
        for c in candidates:
            if c.exists():
                src = str(c)
                break
    print(f"Using source: {src}")
    extract_2025_to_csv(src, out)
