"""Ten questions with a realistic broken first generation and a hand-written gold query.

The broken SQL reproduces error classes seen from LLMs on this schema: SQLite
functions, guessed column and table names, missing GROUP BY, and type mix-ups.
Each question names every column the gold query returns so a correct answer
cannot fail on shape alone.
"""

SEEDS = [
    {
        "question": "How many points per game did Nikola Jokic average in the 2023-24 regular season, rounded to one decimal?",
        "broken": "SELECT ROUND(AVG(points), 1) FROM player_game_log WHERE player_id = 203999 AND season = '2023-24'",
        "gold": "SELECT ROUND(AVG(pts), 1) FROM player_game_log WHERE player_id = 203999 AND season = '2023-24' AND season_type = 'Regular Season'",
    },
    {
        "question": "How many games did the Boston Celtics win at home in the 2023-24 regular season?",
        "broken": "SELECT COUNT(*) FROM games WHERE home_team_id = 1610612738 AND season = '2023-24' AND home_score > away_score",
        "gold": "SELECT COUNT(*) FROM game WHERE home_team_id = 1610612738 AND season = '2023-24' AND season_type = 'Regular Season' AND home_score > away_score",
    },
    {
        "question": "How many regular season games were played in December 2022?",
        "broken": "SELECT COUNT(*) FROM game WHERE strftime('%Y-%m', game_date) = '2022-12' AND season_type = 'Regular Season'",
        "gold": "SELECT COUNT(*) FROM game WHERE game_date >= '2022-12-01' AND game_date < '2023-01-01' AND season_type = 'Regular Season'",
    },
    {
        "question": "What was Stephen Curry's total three-pointers made in the 2015-16 regular season?",
        "broken": "SELECT SUM(three_pm) FROM player_game_log WHERE player_id = 201939 AND season = '2015-16' AND season_type = 'Regular Season'",
        "gold": "SELECT SUM(fg3m) FROM player_game_log WHERE player_id = 201939 AND season = '2015-16' AND season_type = 'Regular Season'",
    },
    {
        "question": "Which team scored the most total points in the 2023-24 regular season, and how many points was it?",
        "broken": "SELECT t.full_name, SUM(l.pts) FROM team_game_log l JOIN team t ON t.team_id = l.team_id WHERE l.season = '2023-24' AND l.season_type = 'Regular Season' ORDER BY 2 DESC LIMIT 1",
        "gold": "SELECT t.full_name, SUM(l.pts) FROM team_game_log l JOIN team t ON t.team_id = l.team_id WHERE l.season = '2023-24' AND l.season_type = 'Regular Season' GROUP BY t.full_name ORDER BY 2 DESC LIMIT 1",
    },
    {
        "question": "How many total rebounds did Giannis Antetokounmpo grab in the 2020-21 playoffs?",
        "broken": "SELECT SUM(rebounds) FROM player_game_log WHERE player_id = 203507 AND season = '2020-21' AND season_type = 'Playoffs'",
        "gold": "SELECT SUM(reb) FROM player_game_log WHERE player_id = 203507 AND season = '2020-21' AND season_type = 'Playoffs'",
    },
    {
        "question": "On what date did LeBron James have his highest-scoring game of the 2022-23 regular season, and how many points did he score?",
        "broken": "SELECT game_date, pts FROM player_game_log WHERE player_name = 'LeBron James' AND season = '2022-23' AND season_type = 'Regular Season' ORDER BY pts DESC LIMIT 1",
        "gold": "SELECT game_date, pts FROM player_game_log WHERE player_id = 2544 AND season = '2022-23' AND season_type = 'Regular Season' ORDER BY pts DESC LIMIT 1",
    },
    {
        "question": "How many regular season games in 2022-23 went to overtime?",
        "broken": "SELECT COUNT(*) FROM game WHERE season = '2022-23' AND season_type = 'Regular Season' AND overtime = true",
        "gold": "SELECT COUNT(DISTINCT game_id) FROM team_game_log WHERE season = '2022-23' AND season_type = 'Regular Season' AND minutes > 240",
    },
    {
        "question": "What was the Denver Nuggets' average plus-minus per game in the 2022-23 playoffs, rounded to one decimal?",
        "broken": "SELECT ROUND(AVG(plus_minus), 1) FROM team_game_log WHERE team_id = 1610612743 AND season = '2022-23' AND season_type = 'Playoffs' GROUP BY",
        "gold": "SELECT ROUND(AVG(plus_minus), 1) FROM team_game_log WHERE team_id = 1610612743 AND season = '2022-23' AND season_type = 'Playoffs'",
    },
    {
        "question": "How many total assists did Luka Doncic have in the 2023-24 regular season?",
        "broken": "SELECT SUM(ast) FROM player_game_log WHERE player_id = 1629029 AND season = '2023-24' AND season_type = 'Regular Season' AND ast > 'ten'",
        "gold": "SELECT SUM(ast) FROM player_game_log WHERE player_id = 1629029 AND season = '2023-24' AND season_type = 'Regular Season'",
    },
]
