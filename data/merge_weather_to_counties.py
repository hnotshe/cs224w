#!/usr/bin/env python3
"""
Merge NOAA county-level weather CSV (with comment lines) into a counties CSV.

- Counties CSV must have columns: county_name, state_abbr (two-letter), plus any others.
- NOAA CSV may start with lines beginning with "#", then a header like:
  ID,Name,State,Value,Rank,Anomaly (1901-2000 base period),1901-2000 Mean

We join on (normalized county_name, state_abbr).
Normalization removes trailing "County", "Parish", etc., and uppercases.

Usage:
  python merge_weather_to_counties.py --counties counties.csv --weather data.csv --out merged.csv --prefix aug_2025_temp_

This will add columns:
  <prefix>value, <prefix>rank, <prefix>anomaly, <prefix>mean1901_2000
"""
import argparse, re, io, sys
import pandas as pd

def smart_read_counties(path: str) -> pd.DataFrame:
    # try to read with common separators
    for sep in [',','\t',';','|']:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return pd.read_csv(path, engine='python')

def read_noaa_county_csv(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    # find start of header (first line that looks like CSV with commas)
    start_idx = 0
    for i, ln in enumerate(lines):
        if ',' in ln:
            start_idx = i
            break
    csv_text = ''.join(lines[start_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    return df

def norm_name(x: str) -> str:
    x = str(x)
    x = re.sub(r'\s+County$','', x, flags=re.I)
    x = re.sub(r'\s+Parish$','', x, flags=re.I)
    x = re.sub(r'\s+City and Borough$','', x, flags=re.I)
    x = re.sub(r'\s+Municipio$','', x, flags=re.I)
    x = re.sub(r'\s+City$','', x, flags=re.I)
    return x.strip().upper()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counties', required=True, help='Path to counties.csv')
    ap.add_argument('--weather', required=True, help='Path to NOAA county weather CSV')
    ap.add_argument('--out', required=True, help='Path to write merged CSV')
    ap.add_argument('--prefix', default='weather_', help='Prefix for weather columns (e.g., "aug_2025_temp_")')
    args = ap.parse_args()

    counties = smart_read_counties(args.counties)
    weather = read_noaa_county_csv(args.weather)

    # validate expected columns exist
    if not {'county_name','state_abbr'}.issubset(set(map(str.lower, counties.columns.str.lower()))):
        # try exact names first
        if 'county_name' not in counties.columns or 'state_abbr' not in counties.columns:
            print("ERROR: counties.csv must contain 'county_name' and 'state_abbr' columns", file=sys.stderr)
            sys.exit(2)

    cnt = counties.copy()
    cnt['county_key'] = cnt['county_name'].map(norm_name)
    cnt['state_key'] = cnt['state_abbr'].str.upper()

    w = weather.copy()
    # derive state two-letter code from ID if present
    if 'ID' in w.columns:
        w['state_key'] = w['ID'].astype(str).str.extract(r'^([A-Z]{2})-')[0]
    if 'state_key' not in w.columns or w['state_key'].isna().all():
        if 'State' in w.columns:
            w['state_key'] = w['State'].astype(str).str[:2].str.upper()
        else:
            print("ERROR: weather csv must contain 'ID' (e.g., AL-001) or 'State' columns", file=sys.stderr)
            sys.exit(3)
    name_col = 'Name' if 'Name' in w.columns else None
    if name_col is None:
        print("ERROR: weather csv must contain 'Name' (county name) column", file=sys.stderr)
        sys.exit(4)
    w['county_key'] = w[name_col].map(norm_name)

    # Select & rename weather columns
    val_col = 'Value' if 'Value' in w.columns else None
    rank_col = 'Rank' if 'Rank' in w.columns else None
    anom_col = None
    mean_col = None
    for c in w.columns:
        if isinstance(c, str):
            if c.lower().startswith('anomaly'):
                anom_col = c
            if '1901-2000' in c:
                mean_col = c

    sel_cols = ['county_key','state_key']
    rename_map = {}
    if val_col: sel_cols.append(val_col); rename_map[val_col] = f'{args.prefix}value'
    if rank_col: sel_cols.append(rank_col); rename_map[rank_col] = f'{args.prefix}rank'
    if anom_col: sel_cols.append(anom_col); rename_map[anom_col] = f'{args.prefix}anomaly'
    if mean_col: sel_cols.append(mean_col); rename_map[mean_col] = f'{args.prefix}mean1901_2000'

    w_small = w[sel_cols].rename(columns=rename_map)

    merged = cnt.merge(w_small, on=['county_key','state_key'], how='left').drop(columns=['county_key','state_key'])
    # Report coverage to stderr
    cov = merged[f'{args.prefix}value'].notna().mean() if f'{args.prefix}value' in merged.columns else float('nan')
    print(f"Merge complete. Coverage of weather value: {cov:.2%}", file=sys.stderr)

    merged.to_csv(args.out, index=False)

if __name__ == '__main__':
    main()
