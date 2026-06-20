"""Generate match predictions from a fitted model."""
import math

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

    best_g1, best_g2 = max(probs, key=probs.__getitem__)

    return {
        "ph": round(ph, 4),
        "pu": round(pu, 4),
        "pb": round(pb, 4),
        "odds_h": round(1.0 / ph, 2) if ph > 0 else None,
        "odds_u": round(1.0 / pu, 2) if pu > 0 else None,
        "odds_b": round(1.0 / pb, 2) if pb > 0 else None,
        "best_score": f"{best_g1}-{best_g2}",
    }


def predict_matches(upcoming: list[dict], model: dict) -> list[dict]:
    results = []
    for m in upcoming:
        pred = predict_match(m["team1"], m["team2"], model)
        results.append(
            {
                "date": m["date"].isoformat(),
                "team1": m["team1"],
                "team2": m["team2"],
                "group": m.get("group", ""),
                **pred,
            }
        )
    return results
