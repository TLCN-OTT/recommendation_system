from fastapi import FastAPI, HTTPException
import pandas as pd
from data.load_auditlog import load_auditlog
from data.load_content import load_content
from models.cf_model import build_cf_predictions
from models.hybrid import hybrid_recommend
from features.text_features import build_tfidf_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import log

app = FastAPI(title="OTT Recommendation API", version="1.0.0")

# Global variables for data (load once on startup for simplicity)
auditlog = None
content = None
user_item = None
R = None
cf_pred = None
tfidf_mx = None
sim_mx = None
item_index = None

@app.on_event("startup")
def load_data():
    global auditlog, content, user_item, R, cf_pred, tfidf_mx, sim_mx, item_index
    try:
        auditlog = load_auditlog()
        content = load_content()
        user_item = auditlog.groupby(["userId", "itemid"])["rating"].max().reset_index()
        R = user_item.pivot_table(index="userId", columns="itemid", values="rating", fill_value=0)
        cf_pred = build_cf_predictions(R)
        tfidf_mx = build_tfidf_matrix(content)
        sim_mx = cosine_similarity(tfidf_mx)
        item_index = {item: i for i, item in enumerate(content["itemid"])}
        log("Data loaded successfully.")
    except Exception as e:
        log(f"Error loading data: {e}")
        raise

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: str, top_n: int = 10):
    if user_id not in R.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        rec = hybrid_recommend(user_id, user_item, cf_pred, sim_mx, content, item_index, top_n)
        # Add type: 'movie' if movieid not null, else 'series'
        rec["type"] = rec["itemid"].map(content.set_index("itemid")["type"])
        result = rec[["itemid", "type"]].to_dict(orient="records")
        return {"user_id": user_id, "recommendations": result}
    except Exception as e:
        log(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)