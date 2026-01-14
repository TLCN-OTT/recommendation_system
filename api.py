from fastapi import FastAPI, HTTPException
import time
import threading

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data.load_auditlog import load_auditlog
from data.load_content import load_content
from models.cf_model import build_cf_predictions
from models.hybrid import hybrid_recommend
from features.text_features import build_tfidf_matrix
from utils.logger import log


# ================= CONFIG =================
DATA_TTL = 15 * 60 
# =========================================


app = FastAPI(
    title="OTT Recommendation API",
    version="1.0.0"
)


# ================= GLOBAL CACHE =================
data_lock = threading.Lock()
last_loaded = 0

auditlog = None
content = None
user_item = None
R = None
cf_pred = None
tfidf_mx = None
sim_mx = None
item_index = None
# ================================================


# ================= DATA LOADER =================
def load_data():
    global auditlog, content, user_item, R
    global cf_pred, tfidf_mx, sim_mx, item_index, last_loaded

    log(" Reloading data...")

    auditlog = load_auditlog()
    content = load_content()

    user_item = (
        auditlog
        .groupby(["userId", "itemid"])["rating"]
        .max()
        .reset_index()
    )

    R = user_item.pivot_table(
        index="userId",
        columns="itemid",
        values="rating",
        fill_value=0
    )

    cf_pred = build_cf_predictions(R)
    tfidf_mx = build_tfidf_matrix(content)
    sim_mx = cosine_similarity(tfidf_mx)

    item_index = {
        item: i for i, item in enumerate(content["itemid"])
    }

    last_loaded = time.time()
    log("Data loaded successfully")


def ensure_data_loaded():
    global last_loaded

    now = time.time()

    if last_loaded == 0 or now - last_loaded > DATA_TTL:
        with data_lock:
            # double check sau khi lock
            if last_loaded == 0 or time.time() - last_loaded > DATA_TTL:
                load_data()
# ================================================


# ================= HEALTH =================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_loaded": last_loaded
    }
# =========================================


# ================= RECOMMEND =================
@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_n: int = 10):
    ensure_data_loaded()

    if R is None or user_id not in R.index:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    try:
        rec = hybrid_recommend(user_id, user_item, cf_pred, sim_mx, content, item_index, top_n) # Add type: 'movie' if movieid not null, else 'series' 
        rec["type"] = rec["itemid"].map(content.set_index("itemid")["type"]) 
        result = rec[["itemid", "type"]].to_dict(orient="records") 
        return {"user_id": user_id, "recommendations": result}

    except Exception as e:
        log(f" Recommend error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
# =========================================


# ================= LOCAL RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
# =========================================
