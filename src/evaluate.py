"""Evaluate past predictions against actual results."""
import math
from datetime import date
from pathlib import Path

from src.history import load_all_predictions


def evaluate(played: list[dict], history_dir: Path) -> dict | None:
    """
    For each played match, find the most recent prediction made before match date
    and compare it against the actual result. Returns metrics or None if no data.
    """
    history = load_all_predictions(history_dir)
    if not history:
        return None

    # Build lookup: (team1, team2, match_date) -> list of (pred_date, prediction)
    pred_by_match: dict[tuple, list[tuple[date, dict]]] = {}
    for entry in history:
        pred_date = date.fromisoformat(entry["date"])
        for pred in entry["predictions"]:
            match_date = date.fromisoformat(pred["date"])
            if pred_date >= match_date:
                # Only count predictions made strictly before the match
                continue
            key = (pred["team1"], pred["team2"], match_date)
            pred_by_match.setdefault(key, []).append((pred_date, pred))

    brier_scores: list[float] = []
    log_losses: list[float] = []
    exact_hits: list[int] = []
    outcome_hits: list[int] = []

    for m in played:
        g1, g2 = m["score"]
        match_date = m["date"]
        key = (m["team1"], m["team2"], match_date)

        candidates = pred_by_match.get(key, [])
        if not candidates:
            continue

        # Use the most recent prediction before match day
        _, pred = max(candidates, key=lambda x: x[0])

        actual_h = 1 if g1 > g2 else 0
        actual_u = 1 if g1 == g2 else 0
        actual_b = 1 if g1 < g2 else 0

        ph, pu, pb = pred["ph"], pred["pu"], pred["pb"]

        brier = ((ph - actual_h) ** 2 + (pu - actual_u) ** 2 + (pb - actual_b) ** 2) / 3
        brier_scores.append(brier)

        p_actual = ph * actual_h + pu * actual_u + pb * actual_b
        if p_actual > 0:
            log_losses.append(-math.log(p_actual))

        predicted_score = pred.get("best_score", "")
        actual_score = f"{g1}-{g2}"
        exact_hits.append(1 if predicted_score == actual_score else 0)

        predicted_outcome = (
            "h" if ph >= pu and ph >= pb else
            "u" if pu >= ph and pu >= pb else
            "b"
        )
        actual_outcome = "h" if g1 > g2 else "u" if g1 == g2 else "b"
        outcome_hits.append(1 if predicted_outcome == actual_outcome else 0)

    n = len(brier_scores)
    if n == 0:
        return None

    return {
        "n_evaluated": n,
        "brier_score": round(sum(brier_scores) / n, 4),
        "log_loss": round(sum(log_losses) / len(log_losses), 4) if log_losses else None,
        "exact_hit_rate": round(sum(exact_hits) / n, 4),
        "outcome_hit_rate": round(sum(outcome_hits) / n, 4),
    }
