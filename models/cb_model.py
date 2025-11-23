from sklearn.metrics.pairwise import cosine_similarity
from config.setting import CB_TOP_SIM

def predict_cb(user_items, target_idx, sim_matrix, item_index):
    sims = []
    for item, rating in user_items:
        idx = item_index.get(item)
        if idx is None:
            continue

        s = sim_matrix[target_idx][idx]
        if s > 0:
            sims.append((s, rating))

    if not sims:
        return 0

    sims = sorted(sims, reverse=True)[:CB_TOP_SIM]
    w_sum = sum(s for s, r in sims)

    return (
        sum(s * r for s, r in sims) / w_sum
        if w_sum > 0 else 0
    )
