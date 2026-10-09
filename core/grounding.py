"""Resolve player and team mentions in a question to database IDs.

Targets the research write-up's "wrong literal" errors: the generator gets
`"Nikola Jokic" -> player.player_id = 203999` instead of guessing a name string.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import psycopg

ALIASES_PATH = Path(__file__).with_name("aliases.json")
MIN_SCORE = 0.5

# Capitalized words that start sentences or name seasons, never a player or team on their own.
STOPWORDS = frozenset(
    """
    a an and are as at average averaged averages best by career compare compared
    did do does during each every finals first for from game games had has have
    highest how in is last list lowest most nba of on or over per playoff playoffs
    points rebounds assists regular season seasons show since the their top versus vs
    was were what when where which who whose with year years
    january february march april may june july august september october november december
    monday tuesday wednesday thursday friday saturday sunday
    """.split()
)

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž0-9][A-Za-zÀ-ÖØ-öø-ÿĀ-ž0-9'.\-]*")

PLAYER_SQL = """
SELECT player_id, full_name,
       greatest(similarity(fold_name(full_name), %(span)s),
                similarity(fold_name(coalesce(last_name, '')), %(span)s)) AS score
FROM player
WHERE fold_name(full_name) %% %(span)s
   OR fold_name(coalesce(last_name, '')) %% %(span)s
ORDER BY score DESC, to_year DESC NULLS LAST, player_id DESC
LIMIT 1
"""

TEAM_SQL = """
SELECT team_id, full_name,
       greatest(similarity(fold_name(full_name), %(span)s),
                similarity(fold_name(coalesce(nickname, '')), %(span)s),
                similarity(fold_name(coalesce(city, '')), %(span)s),
                CASE WHEN lower(abbreviation) = %(span)s THEN 1.0 ELSE 0 END) AS score
FROM team
ORDER BY score DESC, team_id
LIMIT 2
"""

EXACT_SQL = {
    "player": """
        SELECT player_id, full_name FROM player
        WHERE fold_name(full_name) = fold_name(%s)
        ORDER BY to_year DESC NULLS LAST LIMIT 1
    """,
    "team": "SELECT team_id, full_name FROM team WHERE fold_name(full_name) = fold_name(%s) LIMIT 1",
}

ID_COLUMN = {"player": "player.player_id", "team": "team.team_id"}


@dataclass(frozen=True)
class Grounding:
    mention: str
    kind: str
    id: int
    name: str
    score: float
    via_alias: bool = False

    def hint(self) -> str:
        return f'"{self.mention}" -> {ID_COLUMN[self.kind]} = {self.id} ({self.name})'


def fold(text: str) -> str:
    """Lowercase and strip accents, matching the database fold_name()."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, tuple[str, str]]:
    data = json.loads(path.read_text())
    aliases: dict[str, tuple[str, str]] = {}
    for kind, plural in (("player", "players"), ("team", "teams")):
        for alias, name in data.get(plural, {}).items():
            aliases[fold(alias)] = (kind, name)
    return aliases


def _strip_possessive(token: str) -> str:
    return re.sub(r"(?:'s|’s|')$", "", token).strip(".-")


def candidate_spans(question: str, taken: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    """Runs of capitalized words, split at stopwords, skipping spans already resolved by an alias."""
    spans: list[tuple[str, int, int]] = []
    run: list[re.Match] = []

    def flush() -> None:
        if not run:
            return
        start, end = run[0].start(), run[-1].end()
        if not any(start < t_end and t_start < end for t_start, t_end in taken):
            text = " ".join(_strip_possessive(m.group()) for m in run)
            spans.append((text, start, end))
        run.clear()

    for match in TOKEN_RE.finditer(question):
        token = match.group()
        bare = _strip_possessive(token)
        is_name_like = bare[:1].isupper() or (bare[:1].isdigit() and run)
        if is_name_like and fold(bare) not in STOPWORDS and not re.fullmatch(r"\d{4}(-\d{2})?", bare):
            run.append(match)
        else:
            flush()
    flush()
    return spans


def _alias_hits(question: str, aliases: dict[str, tuple[str, str]]) -> list[tuple[str, str, str, int, int]]:
    folded = fold(question)
    hits: list[tuple[str, str, str, int, int]] = []
    taken: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, folded):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            kind, name = aliases[alias]
            hits.append((question[start:end], kind, name, start, end))
            taken.append((start, end))
    return hits


def ground(
    question: str,
    conn: psycopg.Connection,
    aliases: dict[str, tuple[str, str]] | None = None,
    min_score: float = MIN_SCORE,
) -> list[Grounding]:
    aliases = load_aliases() if aliases is None else aliases
    results: list[Grounding] = []
    seen: set[tuple[str, int]] = set()

    def add(item: Grounding) -> None:
        key = (item.kind, item.id)
        if key not in seen:
            seen.add(key)
            results.append(item)

    hits = _alias_hits(question, aliases)
    for mention, kind, name, _, _ in hits:
        row = conn.execute(EXACT_SQL[kind], (name,)).fetchone()
        if row:
            add(Grounding(mention, kind, int(row[0]), row[1], 1.0, via_alias=True))

    taken = [(start, end) for *_, start, end in hits]
    for span, _, _ in candidate_spans(question, taken):
        folded = fold(span)
        player = conn.execute(PLAYER_SQL, {"span": folded}).fetchone()
        teams = conn.execute(TEAM_SQL, {"span": folded}).fetchall()
        team_score = float(teams[0][2]) if teams else 0.0
        player_score = float(player[2]) if player else 0.0
        if max(team_score, player_score) < min_score:
            continue
        # A tie goes to the team: "Boston" is the city far more often than Brandon Boston.
        if team_score >= player_score:
            # Equal top scores mean an ambiguous city ("Los Angeles"); hint every candidate.
            for team_id, name, score in teams:
                if float(score) == team_score:
                    add(Grounding(span, "team", int(team_id), name, team_score))
        else:
            add(Grounding(span, "player", int(player[0]), player[1], player_score))
    return results


def hints_text(groundings: list[Grounding]) -> str:
    return "\n".join(item.hint() for item in groundings)
