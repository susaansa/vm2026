"""Poisson model with Dixon-Coles correction, calibrated via MLE."""
import math
from datetime import date

import numpy as np
from scipy.optimize import minimize

# FIFA World Rankings (snapshot June 2026 — frozen at tournament start).
# Fallback: rank 50 for unlisted teams.
FIFA_RANKINGS: dict[str, int] = {
    "Argentina": 1,
    "France": 2,
    "Spain": 3,
    "England": 4,
    "Brazil": 5,
    "Portugal": 6,
    "Netherlands": 7,
    "Belgium": 8,
    "Germany": 9,
    "Colombia": 10,
    "Morocco": 11,
    "Uruguay": 12,
    "Switzerland": 13,
    "USA": 14,
    "Croatia": 15,
    "Japan": 16,
    "Senegal": 17,
    "Mexico": 18,
    "Denmark": 19,
    "Australia": 21,
    "South Korea": 22,
    "Ecuador": 24,
    "Canada": 25,
    "Turkey": 30,
    "Norway": 31,
    "Austria": 32,
    "Iran": 33,
    "Sweden": 34,
    "Scotland": 36,
    "Czech Republic": 37,
    "Egypt": 38,
    "Tunisia": 39,
    "Serbia": 40,
    "Algeria": 42,
    "Ivory Coast": 43,
    "Qatar": 44,
    "South Africa": 52,
    "Saudi Arabia": 56,
    "Ghana": 57,
    "DR Congo": 60,
    "Iraq": 63,
    "Bosnia & Herzegovina": 66,
    "Paraguay": 68,
    "Jordan": 72,
    "Uzbekistan": 75,
    "Panama": 78,
    "Cape Verde": 83,
    "New Zealand": 96,
    "Haiti": 105,
    "Curaçao": 115,
}

_DEFAULT_RANK = 50
_TIME_DECAY = 0.005  # per day; half-life ≈ 139 days
_REG = 0.5           # L2 regularization toward prior; higher = stays closer to FIFA prior
_ATTACK_SCALE = 0.12  # maps log(50/rank) to attack-strength range


def _rank_to_strength(rank: int) -> float:
    """Convert FIFA rank to attack-strength prior (rank 50 → 0.0)."""
    return math.log(50.0 / rank) * _ATTACK_SCALE


def _build_priors(teams: list[str]) -> dict[str, dict[str, float]]:
    priors = {}
    for t in teams:
        rank = FIFA_RANKINGS.get(t, _DEFAULT_RANK)
        s = _rank_to_strength(rank)
        # defense prior = 0 (neutral), not -s.
        # With defense_prior = -s, we get λ = exp(a_i + a_j) = μ for all matches
        # with no data, making every prediction symmetric — which is wrong.
        priors[t] = {"attack": s, "defense": 0.0}
    return priors


def _dc_tau(g1: int, g2: int, lam: float, mu: float, rho: float) -> float:
    if g1 == 0 and g2 == 0:
        return 1.0 - lam * mu * rho
    if g1 == 1 and g2 == 0:
        return 1.0 + mu * rho
    if g1 == 0 and g2 == 1:
        return 1.0 + lam * rho
    if g1 == 1 and g2 == 1:
        return 1.0 - rho
    return 1.0


def _neg_log_likelihood(
    params: np.ndarray,
    teams: list[str],
    matches: list[dict],
    priors: dict[str, dict[str, float]],
    today: date,
) -> float:
    n = len(teams)
    attack = {t: params[i] for i, t in enumerate(teams)}
    defense = {t: params[n + i] for i, t in enumerate(teams)}
    rho = float(params[2 * n])

    nll = 0.0
    for m in matches:
        g1, g2 = m["score"]
        t1, t2 = m["team1"], m["team2"]
        days_ago = (today - m["date"]).days
        w = math.exp(-_TIME_DECAY * max(days_ago, 0))

        lam = math.exp(attack[t1] - defense[t2])
        mu = math.exp(attack[t2] - defense[t1])
        tau = _dc_tau(g1, g2, lam, mu, rho)

        if tau <= 0:
            return 1e9

        log_p = (
            math.log(tau)
            + g1 * math.log(lam) - lam - math.lgamma(g1 + 1)
            + g2 * math.log(mu) - mu - math.lgamma(g2 + 1)
        )
        nll -= w * log_p

    # L2 regularization toward FIFA-ranking prior
    for t in teams:
        nll += _REG * (attack[t] - priors[t]["attack"]) ** 2
        nll += _REG * (defense[t] - priors[t]["defense"]) ** 2

    return nll


def fit_model(played: list[dict]) -> dict:
    """Fit attack/defense/rho via MLE. Returns model params dict."""
    teams = sorted({m["team1"] for m in played} | {m["team2"] for m in played})
    n = len(teams)
    priors = _build_priors(teams)
    today = date.today()

    x0 = np.array(
        [priors[t]["attack"] for t in teams]
        + [priors[t]["defense"] for t in teams]
        + [0.05]
    )

    # Bounds: attack/defense in [-3, 3], rho in [-0.5, 0.5]
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-0.5, 0.5)]

    result = minimize(
        _neg_log_likelihood,
        x0,
        args=(teams, played, priors, today),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    params = result.x
    return {
        "teams": teams,
        "attack": {t: float(params[i]) for i, t in enumerate(teams)},
        "defense": {t: float(params[n + i]) for i, t in enumerate(teams)},
        "rho": float(params[2 * n]),
        "n_matches": len(played),
        "converged": result.success,
    }


def strength_for(team: str, model: dict, key: str) -> float:
    """Return attack or defense strength, falling back to prior if team unknown."""
    if team in model[key]:
        return model[key][team]
    rank = FIFA_RANKINGS.get(team, _DEFAULT_RANK)
    s = _rank_to_strength(rank)
    return s if key == "attack" else -s
