#!/usr/bin/env python3
"""
Merge Census county population CSV (co-est2024-pop.csv style) into a counties CSV.

- The Census CSV is the "table with row headers in column A..." format exported from Excel.
- We parse the two header rows, keep county rows (those starting with a leading '.'),
  split "County, State" into county + full state name, map to two-letter state code,
  normalize the county name (remove "County/Parish/City and Borough/City/Municipio/Borough/Census Area"),
  and join to your counties.csv on (county_name, state_abbr).

Usage:
  python merge_population_to_counties.py \
    --counties counties.csv \
    --population co-est2024-pop.csv \
    --out merged.csv \
    --prefix pop_

This will add:
  <prefix>2020_base, <prefix>2024_est
"""
import argparse, io, re, sys
import pandas as pd

STATE_MAP = {
"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
"Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
"Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA",
"Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
"New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND",
"Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
"Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
"Puerto Rico":"PR"
}

def smart_read_counties(path: str) -> pd.DataFrame:
    for sep in [',','\t',';','|']:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return pd.read_csv(path, engine='python')

def norm_name(x: str) -> str:
    x = str(x)
    x = re.sub(r'\s+County$','', x, flags=re.I)
    x = re.sub(r'\s+Parish$','', x, flags=re.I)
    x = re.sub(r'\s+City and Borough$','', x, flags=re.I)
    x = re.sub(r'\s+Municipio$','', x, flags=re.I)
    x = re.sub(r'\s+Borough$','', x, flags=re.I)
    x = re.sub(r'\s+Census Area$','', x, flags=re.I)
    x = re.sub(r'\s+City$','', x, flags=re.I)
    return x.strip().upper()

def read_census_pop_table(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, engine='python')
    # find "Geographic Area" header
    idx = raw.index[raw[0].astype(str).str.strip().eq("Geographic Area")]
    if len(idx) != 1:
        raise ValueError("Could not find 'Geographic Area' header row")
    h0 = idx[0]
    h1 = h0 + 1  # years row
    # construct columns
    cols = ["Geographic Area", "2020_base", "2020", "2021", "2022", "2023", "2024"]
    cols = cols[:1 + (raw.shape[1]-1)]  # trim to file width
    df = raw.loc[h1+1:, :len(cols)-1].copy()
    df.columns = cols
    # keep county rows (start with '.')
    df = df[df["Geographic Area"].astype(str).str.startswith(".")].copy()
    df["Geographic Area"] = df["Geographic Area"].astype(str).str.lstrip(".").str.strip()
    # split county, state
    parts = df["Geographic Area"].str.rsplit(",", n=1, expand=True)
    df["county_name_full"] = parts[0].str.strip()
    df["state_full"] = parts[1].str.strip()
    df["state_abbr"] = df["state_full"].map(STATE_MAP)
    df["county_key"] = df["county_name_full"].map(norm_name)
    df["state_key"] = df["state_abbr"]
    # select population columns
    out = df[["county_key","state_key","2020_base","2024"]].copy()
    # clean numbers
    for c in ["2020_base","2024"]:
        out[c] = out[c].astype(str).str.replace(",","", regex=False)
        out.loc[out[c].str.lower().eq("nan"), c] = None
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counties", required=True)
    ap.add_argument("--population", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="pop_")
    args = ap.parse_args()

    counties = smart_read_counties(args.counties)
    pop = read_census_pop_table(args.population)

    cnt = counties.copy()
    if 'county_name' not in cnt.columns or 'state_abbr' not in cnt.columns:
        print("ERROR: counties.csv must have 'county_name' and 'state_abbr' columns", file=sys.stderr)
        sys.exit(2)
    cnt["county_key"] = cnt["county_name"].map(norm_name)
    cnt["state_key"] = cnt["state_abbr"].str.upper()

    merged = cnt.merge(pop.rename(columns={
        "2020_base": f"{args.prefix}2020_base",
        "2024": f"{args.prefix}2024_est"
    }), on=["county_key","state_key"], how="left")\
    .drop(columns=["county_key","state_key"])

    cov = merged[f"{args.prefix}2024_est"].notna().mean()
    print(f"Merge complete. Coverage of {args.prefix}2024_est: {cov:.2%}", file=sys.stderr)

    merged.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
