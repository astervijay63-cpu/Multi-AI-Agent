"""Read-only access to the bundled STUDY-042 clinical dataset."""

import csv
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query


router = APIRouter(tags=["Study dataset"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_DATA_DIR = PROJECT_ROOT / "study_data" / "data"
STUDY_ROOT = PROJECT_ROOT / "study_data"
DOCUMENTS_DIR = STUDY_ROOT / "documents"
RESPONSES_DIR = STUDY_ROOT / "responses"


def _csv_files() -> list[Path]:
    directory = DATA_DIR if DATA_DIR.is_dir() else LEGACY_DATA_DIR
    return sorted(directory.glob("*.csv"))


@lru_cache(maxsize=1)
def _dataset_summary() -> dict:
    """Build an inexpensive inventory without loading entire CSVs into memory."""
    csv_files = _csv_files()
    if not csv_files:
        return {"available": False, "domains": [], "subjects": 0}

    domains = []
    for path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
            row_count = sum(1 for _ in reader)
        domains.append({
            "domain": path.stem,
            "records": row_count,
            "columns": columns,
        })

    subject_count = 0
    dm_path = next((path for path in csv_files if path.name == "DM.csv"), None)
    if dm_path:
        with dm_path.open("r", encoding="utf-8-sig", newline="") as handle:
            subject_count = sum(1 for _ in csv.DictReader(handle))

    return {
        "available": True,
        "study_id": "STUDY-042",
        "subjects": subject_count,
        "domains": domains,
        "documents": sorted(path.name for path in DOCUMENTS_DIR.glob("*.md")),
        "responses": sorted(path.name for path in RESPONSES_DIR.glob("*.json")),
    }


@router.get("/study/summary")
async def study_summary():
    """Return the bundled Study Sentinel dataset inventory."""
    return _dataset_summary()


@router.get("/study/{domain}")
async def study_domain(
    domain: str,
    limit: int = Query(default=100, ge=1, le=500),
):
    """Return a bounded preview from one clinical data domain."""
    directory = DATA_DIR if DATA_DIR.is_dir() else LEGACY_DATA_DIR
    path = directory / f"{domain.upper()}.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown study domain: {domain}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break

    return {"domain": domain.upper(), "count": len(rows), "columns": columns, "rows": rows}
