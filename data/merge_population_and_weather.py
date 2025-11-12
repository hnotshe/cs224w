#!/usr/bin/env python3
"""
Merge counties_with_population.csv and counties_with_weather.csv into a single CSV.

Both files should have the same base columns (county_id, county_name, state_fips, etc.)
and unique columns:
- Population file: pop_2020_base, pop_2024_est
- Weather file: aug_2025_temp_value, aug_2025_temp_rank, aug_2025_temp_anomaly, aug_2025_temp_mean1901_2000

The merge is done on county_id, which should be unique in both files.

Usage:
  python merge_population_and_weather.py \
    --population counties_with_population.csv \
    --weather counties_with_weather.csv \
    --out counties_merged.csv
"""
import argparse
import sys
import pandas as pd

def smart_read_csv(path: str) -> pd.DataFrame:
    """Try to read CSV with common separators."""
    for sep in [',', '\t', ';', '|']:
        try:
            return pd.read_csv(path, sep=sep)
        except Exception:
            continue
    return pd.read_csv(path, engine='python')

def main():
    ap = argparse.ArgumentParser(
        description='Merge counties with population and weather data'
    )
    ap.add_argument(
        '--population',
        required=True,
        help='Path to counties_with_population.csv'
    )
    ap.add_argument(
        '--weather',
        required=True,
        help='Path to counties_with_weather.csv'
    )
    ap.add_argument(
        '--out',
        required=True,
        help='Path to write merged CSV'
    )
    args = ap.parse_args()

    # Read both CSV files
    print(f"Reading population data from {args.population}...", file=sys.stderr)
    pop_df = smart_read_csv(args.population)
    
    print(f"Reading weather data from {args.weather}...", file=sys.stderr)
    weather_df = smart_read_csv(args.weather)

    # Validate that county_id exists in both files
    if 'county_id' not in pop_df.columns:
        print("ERROR: population CSV must have 'county_id' column", file=sys.stderr)
        sys.exit(2)
    
    if 'county_id' not in weather_df.columns:
        print("ERROR: weather CSV must have 'county_id' column", file=sys.stderr)
        sys.exit(3)

    # Identify columns unique to each file
    # Common columns that we'll use from the population file
    common_cols = set(pop_df.columns) & set(weather_df.columns)
    pop_unique_cols = set(pop_df.columns) - common_cols
    weather_unique_cols = set(weather_df.columns) - common_cols

    print(f"Common columns: {len(common_cols)}", file=sys.stderr)
    print(f"Population-only columns: {sorted(pop_unique_cols)}", file=sys.stderr)
    print(f"Weather-only columns: {sorted(weather_unique_cols)}", file=sys.stderr)

    # Merge on county_id
    # Use population file as base, and add weather columns
    merged = pop_df.merge(
        weather_df[['county_id'] + list(weather_unique_cols)],
        on='county_id',
        how='outer'  # Use outer join to keep all counties from both files
    )

    # Report merge statistics
    pop_count = len(pop_df)
    weather_count = len(weather_df)
    merged_count = len(merged)
    
    print(f"\nMerge statistics:", file=sys.stderr)
    print(f"  Population file rows: {pop_count}", file=sys.stderr)
    print(f"  Weather file rows: {weather_count}", file=sys.stderr)
    print(f"  Merged rows: {merged_count}", file=sys.stderr)
    
    # Check coverage
    if 'pop_2024_est' in merged.columns:
        pop_coverage = merged['pop_2024_est'].notna().mean()
        print(f"  Population coverage: {pop_coverage:.2%}", file=sys.stderr)
    
    if 'aug_2025_temp_value' in merged.columns:
        weather_coverage = merged['aug_2025_temp_value'].notna().mean()
        print(f"  Weather coverage: {weather_coverage:.2%}", file=sys.stderr)

    # Check for counties in one file but not the other
    pop_ids = set(pop_df['county_id'].unique())
    weather_ids = set(weather_df['county_id'].unique())
    only_in_pop = pop_ids - weather_ids
    only_in_weather = weather_ids - pop_ids
    
    if only_in_pop:
        print(f"  Warning: {len(only_in_pop)} counties only in population file", file=sys.stderr)
    if only_in_weather:
        print(f"  Warning: {len(only_in_weather)} counties only in weather file", file=sys.stderr)

    # Sort by county_id for consistency
    merged = merged.sort_values('county_id').reset_index(drop=True)

    # Write output
    print(f"\nWriting merged data to {args.out}...", file=sys.stderr)
    merged.to_csv(args.out, index=False)
    print(f"Done! Merged CSV saved to {args.out}", file=sys.stderr)

if __name__ == '__main__':
    main()

