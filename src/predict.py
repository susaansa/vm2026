"""Generate match predictions from a fitted model."""
import math
from datetime import date

from src.model import _dc_tau, strength_for

_MAX_GOALS = 7  # score grid: 0..6 x 0..6


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def score_probabilities(team1: str, team2: str, model: dict) -> dict[tuple[int, int], float]:
    """Return P(score = (g1, g2)) for all scores in the grid."""
    a1 = strength_for(team1, model, "attack")
    d1 = strength_for(team1, model, "defense")
    a2 = strength_for(team2, model, "attack")
    d2 = strength_for(team2, model, "defense")
    rho = model["rho"]

    lam = math.exp(a1 - d2)
    mu = math.exp(a2 - d1)

    probs: dict[tuple[int, int], float] = {}
    for g1 in range(_MAX_GOALS):
        for g2 in range(_MAX_GOALS):
            tau = _dc_tau(g1, g2, lam, mu, rho)
            p = tau * _poisson_pmf(g1, lam) * _poisson_pmf(g2, mu)
            probs[(g1, g2)] = max(p, 0.0)

    # Normalize to account for probability mass above the grid ceiling
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def predict_match(team1: str, team2: str, model: dict) -> dict:
    """Return a prediction dict for one match."""
    probs = score_probabilities(team1, team2, model)

    ph = sum(p for (g1, g2), p in probs.items() if g1 > g2)
    pu = sum(p for (g1, g2), p in probs.items() if g1 == g2)
    pb = sum(p for (g1, g2), p in probs.items() if g1 < g2)

    # Re-normalize H/U/B (rounding artefacts)
    total = ph + pu + pb
    if total > 0:
        ph, pu, pb = ph / total, pu / total, pb / total

    # Statistical mode — most likely single score (often misleading for tipping)
    best_g1, best_g2 = max(probs, key=probs.__getitem__)

    # Tipping recommendation — best score within the most likely outcome.
    # Filters to only H/U/B cells matching the winning outcome before taking max.
    if ph >= pu and ph >= pb:
        outcome_scores = {s: p for s, p in probs.items() if s[0] > s[1]}
    elif pu >= ph and pu >= pb:
        outcome_scores = {s: p for s, p in probs.items() if s[0] == s[1]}
    else:
        outcome_scores = {s: p for s, p in probs.items() if s[0] < s[1]}
    tip_g1, tip_g2 = max(outcome_scores, key=outcome_scores.__getitem__)

    # Top 3 most likely exact scores with individual probabilities
    top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
    top3_list = [
        {"score": f"{g1}-{g2}", "pct": round(p * 100, 1)}
        for (g1, g2), p in top3
    ]

    return {
        "ph": round(ph, 4),
        "pu": round(pu, 4),
        "pb": round(pb, 4),
        "odds_h": round(1.0 / ph, 2) if ph > 0 else None,
        "odds_u": round(1.0 / pu, 2) if pu > 0 else None,
        "odds_b": round(1.0 / pb, 2) if pb > 0 else None,
        "mode_score": f"{best_g1}-{best_g2}",
        "tip_score": f"{tip_g1}-{tip_g2}",
        "top3": top3_list,
    }


def _team_form(team: str, played: list[dict]) -> list[dict]:
    """Return all tournament matches played by this team, as {opp, f, m, date}."""
    results = []
    for m in played:
        if m["team1"] == team:
            results.append({"opp": m["team2"], "f": m["score"][0], "m": m["score"][1], "date": m["date"].isoformat()})
        elif m["team2"] == team:
            results.append({"opp": m["team1"], "f": m["score"][1], "m": m["score"][0], "date": m["date"].isoformat()})
    return results


def predict_matches(upcoming: list[dict], model: dict, played: list[dict] | None = None) -> list[dict]:
    results = []
    for m in upcoming:
        pred = predict_match(m["team1"], m["team2"], model)
        context = {}
        if played is not None:
            context = {
                "team1_form": _team_form(m["team1"], played),
                "team2_form": _team_form(m["team2"], played),
            }
        results.append(
            {
                "date": m["date"].isoformat(),
                "kickoff_utc": m.get("kickoff_utc"),
                "team1": m["team1"],
                "team2": m["team2"],
                "group": m.get("group", ""),
                **pred,
                "context": context,
            }
        )
    return results
