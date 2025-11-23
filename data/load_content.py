import pandas as pd
from data.db import engine

CONTENT_QUERY = """
SELECT 
    c.id AS contentid,
    m.id AS movieid,
    tv.id AS seriesid,
    c.title,
    c.description,
    c.type,
    STRING_AGG(DISTINCT cat."categoryName", ', ') AS genres,
    STRING_AGG(DISTINCT t."tagName", ', ') AS keywords,
    STRING_AGG(DISTINCT a.name, ', ') AS cast,
    STRING_AGG(DISTINCT d.name, ', ') AS director
FROM content c
LEFT JOIN content_category cc ON c.id = cc.content_id
LEFT JOIN category cat ON cc.category_id = cat.id
LEFT JOIN content_tag ct ON c.id = ct.content_id
LEFT JOIN tag t ON ct.tag_id = t.id
LEFT JOIN content_actor ca ON c.id = ca.content_id
LEFT JOIN actor a ON ca.actor_id = a.id
LEFT JOIN content_director cd ON c.id = cd.content_id
LEFT JOIN director d ON cd.director_id = d.id
LEFT JOIN movies m ON c.id = m."content_id"
LEFT JOIN tvseries tv ON c.id = tv."content_id"
WHERE c.type IN ('MOVIE', 'TVSERIES')
GROUP BY c.id, m.id, tv.id
"""

def load_content():
    df = pd.read_sql(CONTENT_QUERY, engine)

    # merge movieid + seriesid
    df["itemid"] = df["movieid"].combine_first(df["seriesid"]).astype(str)

    df = df.dropna(subset=["itemid"])

    return df[['itemid', 'title', 'genres', 'director', 'cast', 'keywords', 'description', 'type']]
