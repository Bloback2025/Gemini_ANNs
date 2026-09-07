"""
Date of Creation: September 6, 2026, 5:09 PM
Path: C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\data_validator_gem.py

Annotations:
- This is a clean rewrite of the legacy data_validator.py script, updated with a precise timestamp to track multiple edits on the same day.
- Updated pathing to point directly to the new Gemini_ANNs project root directory.
- Audits core dataset schema requirements (Date, Open, High, Low, Close) and explicitly checks for multiple tickers to flag where group-by logic is necessary, preventing data leakage.
"""

import sys
import pandas as pd
from pathlib import Path

# Updated path for the new Gemini_ANNs project structure
p = Path(r"C:\Users\loweb\AI_Financial_Sims\Gemini_ANNs\hoxnc_testing_retest1.csv")

try:
    df = pd.read_csv(p, parse_dates=["Date"])
except Exception as e:
    print("ERROR: cannot read", p, e)
    sys.exit(2)

req = ["Date", "Open", "High", "Low", "Close"]
miss = [c for c in req if c not in df.columns]

if miss:
    print("MISSING_COLUMNS:", miss)
    sys.exit(3)

print("ROWS:", len(df))
print("COLUMNS OK")

if "Ticker" in df.columns:
    groups = df["Ticker"].nunique()
    print("TICKERS:", groups)
    if groups > 1:
        print("WARNING: multiple tickers present — group-by needed before shift")