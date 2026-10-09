-- HoopQL product schema. Play-by-play is created here so later ingestion has a
-- place to land; no loader writes it yet.

CREATE TABLE team (
    team_id BIGINT PRIMARY KEY,
    abbreviation TEXT NOT NULL,
    full_name TEXT NOT NULL,
    city TEXT,
    nickname TEXT
);

CREATE TABLE player (
    player_id BIGINT PRIMARY KEY,
    full_name TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    from_year INTEGER,
    to_year INTEGER,
    team_id BIGINT REFERENCES team (team_id)
);

CREATE TABLE game (
    game_id TEXT PRIMARY KEY,
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL CHECK (season_type IN ('Regular Season', 'Playoffs')),
    home_team_id BIGINT NOT NULL REFERENCES team (team_id),
    away_team_id BIGINT NOT NULL REFERENCES team (team_id),
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    CHECK (home_team_id <> away_team_id)
);

CREATE INDEX game_season_idx ON game (season);

CREATE TABLE player_game_log (
    player_id BIGINT NOT NULL REFERENCES player (player_id),
    game_id TEXT NOT NULL REFERENCES game (game_id),
    team_id BIGINT NOT NULL REFERENCES team (team_id),
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL CHECK (season_type IN ('Regular Season', 'Playoffs')),
    minutes NUMERIC,
    pts INTEGER NOT NULL,
    reb INTEGER NOT NULL,
    ast INTEGER NOT NULL,
    stl INTEGER NOT NULL,
    blk INTEGER NOT NULL,
    tov INTEGER NOT NULL,
    pf INTEGER NOT NULL,
    fgm INTEGER NOT NULL,
    fga INTEGER NOT NULL,
    fg_pct NUMERIC,
    fg3m INTEGER NOT NULL,
    fg3a INTEGER NOT NULL,
    fg3_pct NUMERIC,
    ftm INTEGER NOT NULL,
    fta INTEGER NOT NULL,
    ft_pct NUMERIC,
    oreb INTEGER NOT NULL,
    dreb INTEGER NOT NULL,
    plus_minus INTEGER,
    wl TEXT CHECK (wl IN ('W', 'L')),
    is_home BOOLEAN NOT NULL,
    PRIMARY KEY (player_id, game_id)
);

CREATE INDEX player_game_log_player_id_game_date_idx ON player_game_log (player_id, game_date);
CREATE INDEX player_game_log_team_id_game_date_idx ON player_game_log (team_id, game_date);
CREATE INDEX player_game_log_season_idx ON player_game_log (season);

CREATE TABLE team_game_log (
    team_id BIGINT NOT NULL REFERENCES team (team_id),
    game_id TEXT NOT NULL REFERENCES game (game_id),
    opponent_team_id BIGINT NOT NULL REFERENCES team (team_id),
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL CHECK (season_type IN ('Regular Season', 'Playoffs')),
    is_home BOOLEAN NOT NULL,
    wl TEXT CHECK (wl IN ('W', 'L')),
    minutes NUMERIC,
    pts INTEGER NOT NULL,
    reb INTEGER NOT NULL,
    ast INTEGER NOT NULL,
    stl INTEGER NOT NULL,
    blk INTEGER NOT NULL,
    tov INTEGER NOT NULL,
    pf INTEGER NOT NULL,
    fgm INTEGER NOT NULL,
    fga INTEGER NOT NULL,
    fg_pct NUMERIC,
    fg3m INTEGER NOT NULL,
    fg3a INTEGER NOT NULL,
    fg3_pct NUMERIC,
    ftm INTEGER NOT NULL,
    fta INTEGER NOT NULL,
    ft_pct NUMERIC,
    oreb INTEGER NOT NULL,
    dreb INTEGER NOT NULL,
    plus_minus INTEGER,
    PRIMARY KEY (team_id, game_id),
    CHECK (team_id <> opponent_team_id)
);

CREATE INDEX team_game_log_team_id_game_date_idx ON team_game_log (team_id, game_date);
CREATE INDEX team_game_log_season_idx ON team_game_log (season);

-- Column comments are the text schema retrieval shows to the model.
-- player_game_log and game are commented on every column.

COMMENT ON COLUMN game.game_id IS 'NBA game identifier, 10 digits with a leading zero. A prefix of 002 is a regular-season game and 004 is a playoff game.';
COMMENT ON COLUMN game.game_date IS 'Date the game was played.';
COMMENT ON COLUMN game.season IS 'Season label such as 2024-25, the season that started in the fall of 2024.';
COMMENT ON COLUMN game.season_type IS 'Regular Season or Playoffs. Preseason and All-Star games are not loaded.';
COMMENT ON COLUMN game.home_team_id IS 'Team playing in its home arena.';
COMMENT ON COLUMN game.away_team_id IS 'Visiting team.';
COMMENT ON COLUMN game.home_score IS 'Points scored by the home team.';
COMMENT ON COLUMN game.away_score IS 'Points scored by the away team.';

COMMENT ON COLUMN player_game_log.player_id IS 'NBA person id for the player.';
COMMENT ON COLUMN player_game_log.game_id IS 'Game this line belongs to.';
COMMENT ON COLUMN player_game_log.team_id IS 'Team the player appeared for in this game.';
COMMENT ON COLUMN player_game_log.game_date IS 'Date of the game, stored here so a player-and-date lookup does not have to join game.';
COMMENT ON COLUMN player_game_log.season IS 'Season label such as 2024-25, the season that started in the fall of 2024.';
COMMENT ON COLUMN player_game_log.season_type IS 'Regular Season or Playoffs. Preseason and All-Star games are not loaded.';
COMMENT ON COLUMN player_game_log.minutes IS 'Minutes played. 36.5 means 36 minutes and 30 seconds. Null when the source did not record a time.';
COMMENT ON COLUMN player_game_log.pts IS 'Points scored.';
COMMENT ON COLUMN player_game_log.reb IS 'Total rebounds, offensive plus defensive.';
COMMENT ON COLUMN player_game_log.ast IS 'Assists.';
COMMENT ON COLUMN player_game_log.stl IS 'Steals.';
COMMENT ON COLUMN player_game_log.blk IS 'Blocked shots.';
COMMENT ON COLUMN player_game_log.tov IS 'Turnovers.';
COMMENT ON COLUMN player_game_log.pf IS 'Personal fouls.';
COMMENT ON COLUMN player_game_log.fgm IS 'Field goals made, counting both two-point and three-point baskets.';
COMMENT ON COLUMN player_game_log.fga IS 'Field goal attempts.';
COMMENT ON COLUMN player_game_log.fg_pct IS 'Field-goal percentage, made divided by attempted. Null when the player attempted none.';
COMMENT ON COLUMN player_game_log.fg3m IS 'Three-point field goals made.';
COMMENT ON COLUMN player_game_log.fg3a IS 'Three-point field goal attempts.';
COMMENT ON COLUMN player_game_log.fg3_pct IS 'Three-point percentage, three-point makes divided by three-point attempts. Null when the player attempted none.';
COMMENT ON COLUMN player_game_log.ftm IS 'Free throws made.';
COMMENT ON COLUMN player_game_log.fta IS 'Free throw attempts.';
COMMENT ON COLUMN player_game_log.ft_pct IS 'Free-throw percentage, makes divided by attempts. Null when the player attempted none.';
COMMENT ON COLUMN player_game_log.oreb IS 'Offensive rebounds.';
COMMENT ON COLUMN player_game_log.dreb IS 'Defensive rebounds.';
COMMENT ON COLUMN player_game_log.plus_minus IS 'Point differential while this player was on the court. Positive means his team outscored the opponent. Null when the source did not record it.';
COMMENT ON COLUMN player_game_log.wl IS 'W if the player''s team won this game, L if it lost.';
COMMENT ON COLUMN player_game_log.is_home IS 'True when the player''s team was the home team.';

COMMENT ON COLUMN team_game_log.season_type IS 'Regular Season or Playoffs. Preseason and All-Star games are not loaded.';
COMMENT ON COLUMN team_game_log.minutes IS 'Team minutes played. 240 is a regulation game (5 players times 48 minutes). Overtime is longer.';
COMMENT ON COLUMN team_game_log.fg3_pct IS 'Three-point percentage, three-point makes divided by three-point attempts. Null when the team attempted none.';
COMMENT ON COLUMN team_game_log.plus_minus IS 'Point differential for the team in this game, equal to its points minus the opponent''s points. Null when the source did not record it.';

-- Ingestion of this table is deferred until after launch.
CREATE TABLE play_by_play (
    game_id TEXT NOT NULL REFERENCES game (game_id),
    event_num INTEGER NOT NULL,
    period SMALLINT,
    clock TEXT,
    description TEXT,
    team_id BIGINT REFERENCES team (team_id),
    player_id BIGINT REFERENCES player (player_id),
    PRIMARY KEY (game_id, event_num)
);

CREATE TABLE etl_run (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    kind TEXT NOT NULL CHECK (kind IN ('backfill', 'daily')),
    seasons TEXT,
    rows_upserted INTEGER NOT NULL DEFAULT 0,
    rows_changed INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE etl_checkpoint (
    endpoint TEXT NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    row_count INTEGER NOT NULL,
    PRIMARY KEY (endpoint, season, season_type)
);
