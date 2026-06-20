"""Fetch and parse WC 2026 match data from openfootball."""
import re
from datetime import date, datetime, timedelta, timezone

import requests

WORLDCUP_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json"
    "/master/2026/worldcup.json"
)

_PLACEHOLDER_RE = re.compile(r"^[0-9]|^[WL]\d|/")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s+UTC([+-]\d+)")


def _is_placeholder(name: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(name))


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _kickoff_utc(date_str: str, time_str: str | None) -> str | None:
    """Parse '13:00 UTC-6' + date and return full UTC ISO datetime string."""
    if not time_str:
        return None
    m = _TIME_RE.match(time_str)
    if not m:
        return None
    hour, minute, offset = int(m.group(1)), int(m.group(2)), int(m.group(3))
    tz = timezone(timedelta(hours=offset))
    d = datetime.strptime(date_str, "%Y-%m-%d")
    dt_local = datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz)
    dt_utc = dt_local.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%MZ")


def fetch_matches() -> list[dict]:
    response = requests.get(WORLDCUP_URL, timeout=30)
    response.raise_for_status()
    raw = response.json()

    matches = []
    for m in raw.get("matches", []):
        ft = None
        score_block = m.get("score")
        if score_block:
            ft_list = score_block.get("ft")
            if ft_list and len(ft_list) == 2:
                ft = (int(ft_list[0]), int(ft_list[1]))

        matches.append(
            {
                "date": _parse_date(m["date"]),
                "team1": m["team1"],
                "team2": m["team2"],
                "group": m.get("group", ""),
                "score": ft,
                "kickoff_utc": _kickoff_utc(m["date"], m.get("time")),
            }
        )

    return matches


def get_played(matches: list[dict]) -> list[dict]:
    return [m for m in matches if m["score"] is not None]


def get_upcoming(matches: list[dict]) -> list[dict]:
    today = date.today()
    return [
        m
        for m in matches
        if m["score"] is None
        and not _is_placeholder(m["team1"])
        and not _is_placeholder(m["team2"])
        and m["date"] >= today
    ]
