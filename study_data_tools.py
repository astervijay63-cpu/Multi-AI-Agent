"""Backend tools for querying the bundled clinical study CSV domains."""

import csv
import json
import re
from pathlib import Path

from langchain_core.tools import tool

from config.logging_config import setup_logging

logger = setup_logging(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_DATA_DIR = PROJECT_ROOT / "study_data" / "data"
_STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "find", "has", "in",
    "is", "of", "on", "or", "study", "the", "to", "with",
}


def _data_directory() -> Path:
    return DATA_DIR if DATA_DIR.is_dir() else LEGACY_DATA_DIR


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9-]+", value.lower())
        if token not in _STOP_WORDS and len(token) > 1
    }


@tool
async def search_local_study_data(
    query: str = "",
    domain: str = "",
    limit: int = 25,
) -> str:
    """Search the local study CSV records used by MOSAIC analysis.

    Use this before database or live API searches when investigating the
    bundled STUDY-042 dataset. Search terms are matched across every field,
    and the response includes domain counts so agents can reason about the
    complete local evidence set. Set ``domain`` to a CSV name such as DM, AE,
    LB, or VS when narrowing the search.
    """
    limit = max(1, min(limit, 100))
    requested_domain = domain.upper().strip()
    query_tokens = _tokens(query)
    directory = _data_directory()
    paths = sorted(directory.glob("*.csv"))

    if requested_domain:
        paths = [path for path in paths if path.stem.upper() == requested_domain]

    domain_counts = {}
    matches = []
    for path in paths:
        rows = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(row)
                haystack = " ".join(str(value or "") for value in row.values()).lower()
                if not query_tokens or all(token in haystack for token in query_tokens):
                    if len(matches) < limit:
                        matches.append({"domain": path.stem, "record": row})
        domain_counts[path.stem] = len(rows)

    return json.dumps({
        "study_id": "STUDY-042",
        "query": query,
        "domain": requested_domain or None,
        "domain_counts": domain_counts,
        "match_count": len(matches),
        "matches": matches,
    }, indent=2)
