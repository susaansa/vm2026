"""Orchestrate: fetch → fit → predict → history → evaluate → publish."""
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from src.evaluate import evaluate
from src.fetch import fetch_matches, get_played, get_upcoming
from src.history import save_predictions
from src.model import fit_model
from src.predict import predict_matches

ROOT = Path(__file__).parent.parent
HISTORY_DIR = ROOT / "data" / "history"
DATA_JSON = ROOT / "docs" / "data.json"
DATA_JSON.parent.mkdir(parents=True, exist_ok=True)


def _write_data_json(predictions: list[dict], metrics: dict | None, n_matches: int) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "upcoming": predictions,
        "metrics": metrics,
        "model_params_n_matches": n_matches,
    }
    DATA_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {DATA_JSON} ({len(predictions)} kamper)")


def main() -> None:
    print("Henter kampdata...")
    try:
        matches = fetch_matches()
    except Exception as e:
        print(f"FEIL: Kunne ikke hente kampdata: {e}", file=sys.stderr)
        sys.exit(1)

    played = get_played(matches)
    upcoming = get_upcoming(matches)
    print(f"Spilte: {len(played)}, kommende: {len(upcoming)}")

    if not played:
        print("Ingen spilte kamper ennå — skriver tomt datasett.")
        _write_data_json([], None, 0)
        return

    print(f"Kalibrerer modell på {len(played)} kamper...")
    model = fit_model(played)
    if not model["converged"]:
        print("Advarsel: optimalisering konvergerte ikke fullt ut.", file=sys.stderr)

    predictions = predict_matches(upcoming, model)
    print(f"Prediksjon generert for {len(predictions)} kommende kamper.")

    today = date.today()
    history_path = save_predictions(predictions, HISTORY_DIR, today)
    print(f"Historikk lagret: {history_path}")

    metrics = evaluate(played, HISTORY_DIR)
    if metrics:
        print(
            f"Treffsikkerhet: utfall {metrics['outcome_hit_rate']:.1%}, "
            f"Brier {metrics['brier_score']:.3f} ({metrics['n_evaluated']} evaluerte)"
        )
    else:
        print("Ingen historikk å evaluere ennå.")

    _write_data_json(predictions, metrics, model["n_matches"])


if __name__ == "__main__":
    main()
