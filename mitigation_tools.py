#!/usr/bin/env python3
"""
mitigation_tools.py

Utility script for restoration projects:
- init:    Scaffold a job folder with sample daily_log.json
- rename:  Batch-rename photos: JOBID_YYYYMMDD_HHMM_<location>_###.jpg
- report:  Generate a Markdown daily report + append CSV summary

Only uses standard library.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import re
import sys
import shutil
import itertools
import textwrap
import os

# ---------- Helpers ----------

TIMESTAMP_FMT = "%Y-%m-%d"
PHOTO_TIME_FMT = "%Y%m%d_%H%M"

VALID_PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def iter_photos(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_PHOTO_EXT:
            yield p

def mtime_as_dt(p: Path) -> datetime:
    ts = p.stat().st_mtime
    return datetime.fromtimestamp(ts)

def next_seq(existing_names, prefix: str) -> int:
    """
    Parse existing files that already start with prefix and return next sequence number.
    """
    seqs = []
    for name in existing_names:
        m = re.match(rf"^{re.escape(prefix)}_(\d{{3}})\.", name)
        if m:
            seqs.append(int(m.group(1)))
    return (max(seqs) + 1) if seqs else 1

def write_csv_row(csv_path: Path, headers: list, row: dict):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        # Write only allowed keys
        w.writerow({k: row.get(k, "") for k in headers})

# ---------- Commands ----------

def cmd_init(args):
    """
    Create structure:
    jobs/<JOBID>/
      logs/
        daily_log_YYYY-MM-DD.json  (sample)
      photos/raw/
      photos/renamed/
      reports/
    """
    jobid = slugify(args.jobid)
    today = datetime.now().strftime(TIMESTAMP_FMT)

    base = Path("jobs") / jobid
    logs = base / "logs"
    photos_raw = base / "photos" / "raw"
    photos_renamed = base / "photos" / "renamed"
    reports = base / "reports"

    for p in (logs, photos_raw, photos_renamed, reports):
        ensure_dir(p)

    sample = {
        "job_id": jobid,
        "date": today,
        "client": {"name": "Client Name", "site": "171 Langford Rd, Town of Blythewood"},
        "project_manager": "Jon Williams",
        "crew": ["Tech 1", "Tech 2"],
        "weather": {"temp_F": 74, "rh_pct": 55},
        "activities": [
            "Set containment with zipper doors in affected rooms.",
            "Installed air scrubbers in affected and general cleaning rooms.",
            "HEPA vacuumed exterior walls; applied antimicrobial spray; sealed with plastic."
        ],
        "equipment": [
            {"type": "Dehumidifier LGR", "id": "D-01", "location": "Room A", "amps": 7.4},
            {"type": "Air Scrubber", "id": "AS-02", "location": "Hallway", "hours": 6.0}
        ],
        "moisture_readings": [
            {"location": "Exterior wall A", "material": "Drywall", "reading": 16},
            {"location": "Baseboard east", "material": "Wood", "reading": 12}
        ],
        "communications": [
            "Spoke with GC about rooftop exhaust motors; electrician scheduled.",
            "Emailed client daily status and photos."
        ],
        "safety": "All crew wore PPE (respirators, gloves, eye protection). No incidents.",
        "notes": "Motors on rooftop not functional; waiting on parts before final cleaning.",
        "photos_dir": str(photos_renamed)
    }

    sample_path = logs / f"daily_log_{today}.json"
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)

    print(f"Initialized job at: {base}")
    print(f"Sample log: {sample_path}")
    print(f"Drop raw photos in: {photos_raw}\nThen run 'rename' to standardize filenames into {photos_renamed}.")

def cmd_rename(args):
    """
    Rename photos to: JOBID_YYYYMMDD_HHMM_<location>_###.<ext>
    Uses file modified time (no EXIF dependency).
    """
    jobid = slugify(args.jobid)
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve() if args.dst else src
    location = slugify(args.location or "general")
    date_override = args.date  # YYYY-MM-DD or None

    if not src.exists():
        print(f"Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(dst)

    # Track existing to compute next sequence per minute prefix
    existing = {p.name for p in dst.iterdir()} if dst.exists() else set()

    moved = 0
    for p in iter_photos(src):
        dt = mtime_as_dt(p)
        if date_override:
            # Keep the time from mtime, but set date to override
            try:
                d = datetime.strptime(date_override, TIMESTAMP_FMT)
                dt = dt.replace(year=d.year, month=d.month, day=d.day)
            except ValueError:
                print("Invalid --date; expected YYYY-MM-DD", file=sys.stderr)
                sys.exit(2)

        stamp = dt.strftime(PHOTO_TIME_FMT)  # e.g., 20251101_1427
        prefix = f"{jobid}_{stamp}_{location}"
        seq = next_seq(existing, prefix)
        new_name = f"{prefix}_{seq:03d}{p.suffix.lower()}"
        new_path = dst / new_name

        # Avoid collisions
        while new_path.exists():
            seq += 1
            new_name = f"{prefix}_{seq:03d}{p.suffix.lower()}"
            new_path = dst / new_name

        shutil.copy2(p, new_path) if args.copy else shutil.move(p, new_path)
        existing.add(new_name)
        moved += 1
        if not args.quiet:
            print(f"{'Copied' if args.copy else 'Moved'} -> {new_path.relative_to(dst.parent)}")

    print(f"Done. {moved} file(s) {'copied' if args.copy else 'moved'} to {dst}")

def _md_section(title: str) -> str:
    return f"\n## {title}\n"

def _md_list(items):
    return "".join([f"- {i}\n" for i in items])

def _md_kv_table(rows):
    # rows: list of (key, value)
    lines = ["| Field | Value |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"

def cmd_report(args):
    """
    Build a Markdown report for a given daily_log.json and append CSV summary.
    """
    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    with log_path.open("r", encoding="utf-8") as f:
        log = json.load(f)

    # Derive paths
    jobid = slugify(str(log.get("job_id", args.jobid or "unknown")))
    date = log.get("date") or datetime.now().strftime(TIMESTAMP_FMT)
    base = Path("jobs") / jobid
    reports_dir = base / "reports"
    ensure_dir(reports_dir)

    md_out = reports_dir / f"{date}_daily_report.md"
    csv_out = reports_dir / "summary.csv"

    # Compose Markdown
    title = f"# Daily Report — Job {jobid} — {date}\n"
    header = _md_kv_table([
        ("Client", log.get("client", {}).get("name", "")),
        ("Site", log.get("client", {}).get("site", "")),
        ("Project Manager", log.get("project_manager", "")),
        ("Crew", ", ".join(log.get("crew", []))),
        ("Weather", f'{log.get("weather", {}).get("temp_F","")}°F / {log.get("weather", {}).get("rh_pct","")}% RH'),
    ])

    activities = _md_section("Activities") + _md_list(log.get("activities", []))

    equip_rows = ["| Type | ID | Location | Notes |", "|---|---|---|---|"]
    for e in log.get("equipment", []):
        notes = []
        for k in ("amps", "hours", "setpoint", "cfm"):
            if k in e:
                notes.append(f"{k}: {e[k]}")
        equip_rows.append(f'| {e.get("type","")} | {e.get("id","")} | {e.get("location","")} | {"; ".join(notes)} |')
    equipment = _md_section("Equipment") + ("\n".join(equip_rows) + "\n" if len(equip_rows) > 2 else "_None_\n")

    moist_rows = ["| Location | Material | Reading |", "|---|---|---|"]
    for m in log.get("moisture_readings", []):
        moist_rows.append(f'| {m.get("location","")} | {m.get("material","")} | {m.get("reading","")} |')
    moisture = _md_section("Moisture Readings") + ("\n".join(moist_rows) + "\n" if len(moist_rows) > 2 else "_None_\n")

    comms = _md_section("Communications") + _md_list(log.get("communications", []))
    safety = _md_section("Safety") + (log.get("safety") or "_None_") + "\n"
    notes = _md_section("Notes & Next Steps") + (log.get("notes") or "_None_") + "\n"

    # Photos
    photos_dir = Path(log.get("photos_dir") or (base / "photos" / "renamed"))
    md_photos = _md_section("Photos")
    photo_lines = []
    if photos_dir.exists():
        # Group by minute-stamp for brief captions
        files = sorted([p for p in iter_photos(photos_dir)], key=lambda p: p.name)
        for p in files:
            rel = p.relative_to(reports_dir.parent) if reports_dir.parent in p.parents else p
            # Basic caption from filename
            cap = re.sub(r"[_-]+", " ", p.stem)
            photo_lines.append(f"![{cap}]({rel.as_posix()})")
    md_photos += ("\n".join(photo_lines) + "\n") if photo_lines else "_No photos found_\n"

    full_md = title + header + activities + equipment + moisture + comms + safety + notes + md_photos

    with md_out.open("w", encoding="utf-8") as f:
        f.write(full_md)

    # Append to CSV summary
    csv_headers = [
        "date", "job_id", "client_name", "site", "pm", "crew_count",
        "weather_tempF", "weather_rh", "activities_count", "equipment_count",
        "moisture_count", "photos_count", "notes"
    ]
    row = {
        "date": date,
        "job_id": jobid,
        "client_name": log.get("client", {}).get("name", ""),
        "site": log.get("client", {}).get("site", ""),
        "pm": log.get("project_manager", ""),
        "crew_count": len(log.get("crew", [])),
        "weather_tempF": log.get("weather", {}).get("temp_F", ""),
        "weather_rh": log.get("weather", {}).get("rh_pct", ""),
        "activities_count": len(log.get("activities", [])),
        "equipment_count": len(log.get("equipment", [])),
        "moisture_count": len(log.get("moisture_readings", [])),
        "photos_count": len(list(iter_photos(photos_dir))) if photos_dir.exists() else 0,
        "notes": (log.get("notes") or "").strip()[:120],
    }
    write_csv_row(csv_out, csv_headers, row)

    print(f"Markdown report: {md_out}")
    print(f"CSV summary updated: {csv_out}")

# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(
        description="Restoration job utilities (init, rename, report).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # 1) Start a new job scaffold
          python mitigation_tools.py init PRJ-24-1883

          # 2) Rename photos based on file modified time
          python mitigation_tools.py rename PRJ-24-1883 jobs/PRJ-24-1883/photos/raw --dst jobs/PRJ-24-1883/photos/renamed --location "Unit-504"

          # 3) Generate today's report from the sample log
          python mitigation_tools.py report jobs/PRJ-24-1883/logs/daily_log_2025-11-01.json

          # 4) Rename but keep originals (copy)
          python mitigation_tools.py rename PRJ-24-1883 ./camera-dump --dst jobs/PRJ-24-1883/photos/renamed --copy
        """)
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Scaffold a job folder with sample daily_log.json")
    sp.add_argument("jobid", help="Job ID (e.g., 24-1883-F-Beaufort)")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("rename", help="Batch-rename photos to standard convention")
    sp.add_argument("jobid", help="Job ID, becomes the filename prefix")
    sp.add_argument("src", help="Folder containing photos to rename/move")
    sp.add_argument("--dst", help="Destination folder (default: src)")
    sp.add_argument("--location", help="Location tag (e.g., Unit-504, Basement)")
    sp.add_argument("--date", help="Override date (YYYY-MM-DD); keeps original time")
    sp.add_argument("--copy", action="store_true", help="Copy instead of move")
    sp.add_argument("--quiet", action="store_true", help="Less verbose output")
    sp.set_defaults(func=cmd_rename)

    sp = sub.add_parser("report", help="Generate Markdown report + CSV summary")
    sp.add_argument("log", help="Path to daily_log.json")
    sp.add_argument("--jobid", help="Override job ID if missing in log")
    sp.set_defaults(func=cmd_report)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
