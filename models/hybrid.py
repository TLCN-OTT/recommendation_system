import pandas as pd
from config.setting import CF_WEIGHT, CB_WEIGHT, POP_WEIGHT
from models.cb_model import predict_cb
from utils.logger import log

def hybrid_recommend(
    user_id, 
    user_item_df, 
    cf_predictions, 
    sim_matrix, 
    items_df, 
    item_index, 
    top_n=10,
    alpha=None,
    beta=None,
    gamma=None
):
    """
    Generate hybrid recommendations combining CF, CB, and Popularity
    
    Args:
        alpha: CF weight (default from config)
        beta: CB weight (default from config)
        gamma: Popularity weight (default from config)
    """
    # Use config values if not specified
    if alpha is None:
        alpha = CF_WEIGHT
    if beta is None:
        beta = CB_WEIGHT
    if gamma is None:
        gamma = POP_WEIGHT
    
    seen = set(user_item_df[user_item_df["userId"] == user_id]["itemid"])
    candidates = [i for i in items_df["itemid"] if i not in seen]
    
    if not candidates:
        log(f"No unseen candidates for user {user_id}.")
        return pd.DataFrame(columns=["itemid", "cf_score", "cb_score", "pop_score", "hybrid_score", "title"])
    
    user_interactions = list(
        user_item_df[user_item_df["userId"] == user_id][["itemid", "rating"]].itertuples(index=False, name=None)
    )
    
    # Calculate popularity score (normalized interaction count)
    item_popularity = user_item_df.groupby("itemid").size().to_dict()
    max_pop = max(item_popularity.values()) if item_popularity else 1
    items_df["pop_score"] = items_df["itemid"].map(lambda x: item_popularity.get(x, 0) / max_pop)
    
    recs = []
    for item in candidates:
        if item not in cf_predictions.columns:
            continue
        
        cf_score = float(cf_predictions.loc[user_id, item])
        
        idx = item_index.get(item)
        if idx is None:
            continue
        cb_score = predict_cb(user_interactions, idx, sim_matrix, item_index)
        
        pop_score = items_df.loc[items_df["itemid"] == item, "pop_score"].values[0]
        
        # Use custom weights
        score = alpha * cf_score + beta * cb_score + gamma * pop_score
        
        recs.append((item, cf_score, cb_score, pop_score, score))
    
    df = pd.DataFrame(recs, columns=["itemid", "cf_score", "cb_score", "pop_score", "hybrid_score"])
    df = df.merge(items_df[["itemid", "title"]], on="itemid", how="left")
    
    log(f"Generated {len(df)} recommendations for user {user_id}.")
    return df.sort_values("hybrid_score", ascending=False).head(top_n)