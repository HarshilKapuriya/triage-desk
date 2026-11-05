"""Deterministic extraction of 20,000 January 2024 CFPB records.

Selection Rule:
1. Source: data/CCDB_Export_5_September_2023_through_March_2024.csv
   (SHA-256: 7a8645f6f602762fcf8cdbf01005aaad9a5834debc9cab549ff94b6b128b1ab0).
2. Filter: 'Date received' in range [2024-01-01 .. 2024-01-31] AND non-empty 'Consumer complaint narrative'.
3. Population in January 2024: 43,920 eligible records across all 31 days.
4. Stratified temporal sampling:
   - For each date d in sorted order:
     - Sort all eligible records on date d by integer 'Complaint ID' ascending (deterministic tie-breaker).
     - Target daily quota k = round(N_d * 20000 / 43920).
     - Select indices floor(i * N_d / k) for i in 0 .. k-1.
   - Slice exactly to 20,000 records.
5. Preserves all 31 dates proportionally, with zero day-boundary truncation.
6. Saves to data/cfpb_complaints_jan2024.csv.
"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def extract_records(
    source_path: str = "data/CCDB_Export_5_September_2023_through_March_2024.csv",
    output_path: str = "data/cfpb_complaints_jan2024.csv",
    target_count: int = 20000,
) -> dict:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # 1. Stream scan into daily buckets
    records_by_day = defaultdict(list)
    fieldnames = None
    total_scanned = 0

    with open(source, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            total_scanned += 1
            date = row.get("Date received", "").strip()
            narrative = row.get("Consumer complaint narrative", "").strip()
            if date.startswith("2024-01-") and narrative:
                records_by_day[date].append(row)

    total_eligible = sum(len(v) for v in records_by_day.values())
    print(f"Total scanned: {total_scanned}")
    print(f"Total eligible January 2024 records: {total_eligible}")

    # 2. Deterministic systematic sampling per day
    selected = []
    for date in sorted(records_by_day.keys()):
        day_rows = sorted(records_by_day[date], key=lambda r: int(r["Complaint ID"]))
        k = round(len(day_rows) * target_count / total_eligible)
        step = len(day_rows) / k
        day_selected = [day_rows[int(i * step)] for i in range(k)]
        selected.extend(day_selected)

    selected = selected[:target_count]
    assert len(selected) == target_count, f"Selected {len(selected)} != target {target_count}"

    # 3. Write output CSV
    out = Path(output_path)
    with open(out, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    # 4. Hash output file
    sha256 = hashlib.sha256()
    with open(out, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    out_hash = sha256.hexdigest()

    print(f"Wrote {len(selected)} records to {output_path} (SHA-256: {out_hash})")
    return {
        "source_rows": total_scanned,
        "eligible_rows": total_eligible,
        "extracted_rows": len(selected),
        "output_path": str(out),
        "output_hash": out_hash,
    }


if __name__ == "__main__":
    extract_records()

