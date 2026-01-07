# src/data_loader.py
from __future__ import annotations

import os
import re
from typing import Optional, List

import numpy as np
import pandas as pd


def load_excel(path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_first_csv_in_dir(directory: str = ".") -> pd.DataFrame:
    files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV file found in directory: {directory}")
    csv_path = os.path.join(directory, files[0])
    return pd.read_csv(csv_path, on_bad_lines="skip", sep=None, engine="python")


def convert_runtime(time_str) -> int:
    if pd.isna(time_str):
        return 100
    try:
        hours = re.search(r"(\d+)\s*hr", str(time_str))
        minutes = re.search(r"(\d+)\s*min", str(time_str))
        h = int(hours.group(1)) if hours else 0
        m = int(minutes.group(1)) if minutes else 0
        val = h * 60 + m
        return val if val > 0 else 100
    except Exception:
        return 100


def clean_genre_list(x) -> List[str]:
    if pd.isna(x) or not isinstance(x, str):
        return ["Unknown"]
    clean_s = x.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    out = [g.strip() for g in clean_s.split(",") if g.strip()]
    return out if out else ["Unknown"]


def parse_release_date(series: pd.Series) -> pd.Series:
    """
    Parses dates by trying specific formats first to avoid UserWarnings.
    """
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"]
    for fmt in fmts:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.7:
            return parsed
    # Fallback with dayfirst=True to handle European formats and suppress warnings
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def clean_and_engineer_movies_df(df: pd.DataFrame, target_genres: Optional[List[str]] = None) -> pd.DataFrame:
    if target_genres is None:
        target_genres = ["Action", "Adventure", "Animation", "Comedy", "Drama", "Family", "Sci-Fi", "Horror", "Thriller"]

    d = df.copy()

    # Column existence check 
    if "Title" not in d.columns:
        d["Title"] = np.arange(len(d))
    if "Distributor" not in d.columns:
        d["Distributor"] = "Unknown"
    if "Genre" not in d.columns:
        d["Genre"] = "Unknown"
    if "Release Date" not in d.columns:
        d["Release Date"] = pd.NaT
    if "Running Time" not in d.columns:
        d["Running Time"] = np.nan

    # Numeric columns fallbacks
    for col in ["Budget (in $)", "World Wide Sales (in $)", "Domestic Opening (in $)", "Domestic Sales (in $)"]:
        if col not in d.columns:
            d[col] = np.nan

    # Data Cleaning
    # Runtime
    d["Running Time"] = d["Running Time"].apply(convert_runtime)

    # Genres
    d["Genre_List"] = d["Genre"].apply(clean_genre_list)
    d["Main_Genre"] = d["Genre_List"].apply(lambda x: x[0] if isinstance(x, list) and len(x) else "Unknown")
    for g in target_genres:
        d[f"is_{g}"] = d["Genre_List"].apply(lambda xs: 1 if isinstance(xs, list) and g in xs else 0)

    #Date Processing
    # Dates
    d["Release Date"] = parse_release_date(d["Release Date"])
    d["Year"] = d["Release Date"].dt.year
    d["Release_Month"] = d["Release Date"].dt.month.fillna(0).astype(int)

    # Numeric conversions
    for col in ["Budget (in $)", "World Wide Sales (in $)", "Domestic Opening (in $)", "Domestic Sales (in $)"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Median filling 
    for col in ["Budget (in $)", "World Wide Sales (in $)", "Domestic Opening (in $)"]:
        if d[col].notna().any():
            d[col] = d[col].fillna(d[col].median())
        else:
            d[col] = d[col].fillna(0)

    d["Domestic Sales (in $)"] = d["Domestic Sales (in $)"].fillna(0)

    # ROI
    d["ROI"] = np.where(d["Budget (in $)"] > 0, d["World Wide Sales (in $)"] / d["Budget (in $)"], np.nan)

    # International Sales
    d["International Sales (in $)"] = d["World Wide Sales (in $)"] - d["Domestic Sales (in $)"]
    d.loc[d["International Sales (in $)"].isna(), "International Sales (in $)"] = d["World Wide Sales (in $)"] * 0.4

    return d


def clean_for_timeseries_lstm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    date_col = next((c for c in d.columns if "date" in c.lower()), None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d["Release_Month"] = d[date_col].dt.month
    else:
        d["Release_Month"] = 6

    for col in ["World Wide Sales (in $)", "Budget (in $)", "Domestic Opening (in $)"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d.dropna(subset=["World Wide Sales (in $)", "Budget (in $)"])

    d["Log_Sales"] = np.log1p(d["World Wide Sales (in $)"])
    d["Log_Budget"] = np.log1p(d["Budget (in $)"])

    d["Release_Month"] = pd.to_numeric(d["Release_Month"], errors="coerce").fillna(6).astype(int)
    d.loc[(d["Release_Month"] < 1) | (d["Release_Month"] > 12), "Release_Month"] = 6

    return d