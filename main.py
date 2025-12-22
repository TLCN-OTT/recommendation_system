import pandas as pd
from data.load_auditlog import load_auditlog
from data.load_content import load_content
from models.cf_model import build_cf_predictions
from models.hybrid import hybrid_recommend
from features.text_features import build_tfidf_matrix
from utils.timer import Timer
from utils.logger import log
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == "__main__":
    try:
        with Timer("Load data"):
            auditlog = load_auditlog()
            content = load_content()
        
        with Timer("Prepare rating matrix"):
            user_item = auditlog.groupby(["userId", "itemid"])["rating"].max().reset_index()
            R = user_item.pivot_table(index="userId", columns="itemid", values="rating", fill_value=0)
        
        with Timer("Collaborative Filtering"):
            cf_pred = build_cf_predictions(R)  # Still computed for potential use, but not passed to hybrid
        
        with Timer("Content-Based TF-IDF"):
            tfidf_mx = build_tfidf_matrix(content)
            sim_mx = cosine_similarity(tfidf_mx)
        
        item_index = {item: i for i, item in enumerate(content["itemid"])}
        
        uid = user_item["userId"].iloc[0]
        
        with Timer("Hybrid Recommendation"):
            # In main.py, change the hybrid call back to:
            rec = hybrid_recommend(uid, user_item, cf_pred, sim_mx, content, item_index)
        print(rec[["title", "cf_score", "cb_score", "pop_score", "hybrid_score"]])
        
    except Exception as e:
        log(f"Error in main.py: {e}")
        raise