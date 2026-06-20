"""Save and load date-stamped prediction history files."""
import json
from datetime import date
from pathlib import Path


def save_predictions(predictions: list[dict], history_dir: Path, run_date: date) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{run_date.isoformat()}.json"
    payload = {"date": run_date.isoformat(), "predictions": predictions}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_all_predictions(history_dir: Path) -> list[dict]:
    """Return all history entries sorted oldest-first."""
    if not history_dir.exists():
        return []
    entries = []
    for p in sorted(history_dir.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        entries.append(data)
    return entries
