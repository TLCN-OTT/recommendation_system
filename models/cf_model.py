# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from config.setting import TOP_K_SIMILAR_USERS
# from utils.logger import log

# def build_cf_predictions(rating_matrix):
#     user_mean = rating_matrix.mean(axis=1)
#     norm = rating_matrix.sub(user_mean, axis=0)

#     sim = cosine_similarity(norm)
#     sim_df = pd.DataFrame(sim, index=rating_matrix.index, columns=rating_matrix.index)

#     predictions = pd.DataFrame(index=rating_matrix.index, columns=rating_matrix.columns)

#     for user in rating_matrix.index:
#         sims = sim_df[user].drop(user)
#         sims = sims[sims > 0].sort_values(ascending=False).head(TOP_K_SIMILAR_USERS)

#         for item in rating_matrix.columns:
#             relevant = rating_matrix.loc[sims.index, item]
#             rated = relevant[relevant > 0]

#             if rated.empty:
#                 predictions.loc[user, item] = user_mean[user]
#             else:
#                 w = sims[rated.index]
#                 predictions.loc[user, item] = (rated * w).sum() / w.sum()

#     return predictions.fillna(0)
import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from config.setting import ALS_FACTORS, ALS_REGULARIZATION, ALS_ITERATIONS
from utils.logger import log

def build_cf_predictions(rating_matrix):
    # Convert to sparse matrix (implicit expects users x items)
    sparse_matrix = csr_matrix(rating_matrix.values)
    
    # Initialize ALS model
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=42
    )
    
    # Fit the model
    model.fit(sparse_matrix)
    
    # Generate predictions for all users and items
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    predictions = pd.DataFrame(
        np.dot(user_factors, item_factors.T),
        index=rating_matrix.index,
        columns=rating_matrix.columns
    )
    
    log("CF predictions built using ALS.")
    return predictions