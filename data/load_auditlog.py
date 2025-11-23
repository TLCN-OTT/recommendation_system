import pandas as pd
from data.db import engine
from features.extract_ids import parse_ids

ACTION_TO_RATING = {
    'PLAY_MOVIE': 5,
    'PLAY_EPISODE_OF_SERIES': 5,
    'LIKE_SERIES': 4,
    'LIKE_MOVIE': 4,
    'CREATE_REVIEW': 4,
    'ADD_MOVIE_TO_WATCHLIST': 3,
    'ADD_SERIES_TO_WATCHLIST': 3,
}

def load_auditlog():
    df = pd.read_sql(
        "SELECT \"userId\", action, description FROM audit_logs", 
        engine
    )
    df["itemid"] = df.apply(parse_ids, axis=1)
    df = df.dropna(subset=["itemid"])

    df["rating"] = df["action"].map(ACTION_TO_RATING)
    df = df.dropna(subset=["rating"])

    return df[['userId', 'itemid', 'rating']]
