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
