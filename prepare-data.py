import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

# ---------------------
# 1️⃣ Kết nối PostgreSQL
# ---------------------
engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# ---------------------
# 2️⃣ Lấy auditlog
# ---------------------
auditlog = pd.read_sql("SELECT \"userId\", action, description FROM audit_logs", engine)
print(f"📊 Total audit logs: {len(auditlog)}")
print(f"📊 Unique users: {auditlog['userId'].nunique()}")
print(f"📊 Actions: {auditlog['action'].value_counts().to_dict()}")

# ---------------------
# 3️⃣ Hàm extract movieId và tvSeriesId - FIXED
# ---------------------
def parse_ids(row):
    desc = str(row['description'])
    
    # Pattern cho UUID
    match = re.search(r'with ID ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', desc, re.IGNORECASE)
    if not match:
        return None
    
    # ALL IDs are movieId or seriesId, no contentId
    return str(match.group(1))

auditlog['itemid'] = auditlog.apply(parse_ids, axis=1)

print(f"\n🔍 Extracted IDs:")
print(f"  - Total IDs extracted: {auditlog['itemid'].notna().sum()}")
print(f"  - Unique items: {auditlog['itemid'].nunique()}")

# ---------------------
# 4️⃣ Lấy bảng content - SIMPLIFIED QUERY
# ---------------------
query = """
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

content_df = pd.read_sql(query, engine)
print(f"\n📚 Content loaded:")
print(f"  - Total content: {len(content_df)}")
print(f"  - Movies: {content_df['movieid'].notna().sum()}")
print(f"  - TV Series: {content_df['seriesid'].notna().sum()}")

# Tạo mapping dictionaries - NOT NEEDED ANYMORE
# All IDs are already movieId/seriesId
print(f"\n✅ All IDs are movieId/seriesId - no mapping needed")

# ---------------------
# 5️⃣ Filter valid items - SIMPLIFIED
# ---------------------
print(f"\n✅ Items validation:")
print(f"  - Before: {len(auditlog)} logs")

# Remove logs without itemid
auditlog = auditlog.dropna(subset=['itemid'])
print(f"  - After: {len(auditlog)} logs with valid itemid")
print(f"  - Unique items: {auditlog['itemid'].nunique()}")

# Debug: Check distribution
print(f"\n📊 Item types in auditlog:")
by_action = auditlog.groupby('action')['itemid'].nunique()
print(by_action.to_dict())

if auditlog.empty:
    print("❌ ERROR: No valid itemid found!")
    exit()

# ---------------------
# 6️⃣ Action to Rating
# ---------------------
action2rating = {
    'PLAY_MOVIE': 5,
    'PLAY_EPISODE_OF_SERIES': 5,
    'LIKE_SERIES': 4,
    'LIKE_MOVIE': 4,
    'CREATE_REVIEW': 4,
    'ADD_MOVIE_TO_WATCHLIST': 3,
    'ADD_SERIES_TO_WATCHLIST': 3,
}

auditlog['rating'] = auditlog['action'].map(action2rating)
auditlog = auditlog.dropna(subset=['rating'])

print(f"\n⭐ Ratings assigned:")
print(f"  - Total interactions: {len(auditlog)}")
print(auditlog['rating'].value_counts().to_dict())

# Aggregate: lấy max rating cho mỗi user-item
user_item_ratings = auditlog.groupby(['userId', 'itemid'])['rating'].max().reset_index()
print(f"  - Unique user-item pairs: {len(user_item_ratings)}")

# ---------------------
# 7️⃣ Rating Matrix
# ---------------------
rating_matrix = user_item_ratings.pivot_table(
    index='userId',
    columns='itemid',
    values='rating',
    fill_value=0
)

print(f"\n📊 Rating Matrix:")
print(f"  - Shape: {rating_matrix.shape} (users x items)")
print(f"  - Non-zero entries: {(rating_matrix > 0).sum().sum()}")
print(f"  - Sparsity: {(rating_matrix == 0).sum().sum() / rating_matrix.size * 100:.2f}%")

# ---------------------
# 8️⃣ Collaborative Filtering - IMPROVED
# ---------------------
def collaborative_filtering_predict(rating_matrix, k=20):
    """
    User-based CF với cosine similarity thay vì SVD
    """
    # Normalize by user mean
    user_ratings_mean = rating_matrix.mean(axis=1)
    ratings_normalized = rating_matrix.sub(user_ratings_mean, axis=0).fillna(0)
    
    # User similarity matrix (user x user)
    user_similarity = cosine_similarity(ratings_normalized)
    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)
    
    # Predict ratings
    predictions = pd.DataFrame(index=rating_matrix.index, columns=rating_matrix.columns)
    
    for user in rating_matrix.index:
        # Get similar users (exclude self)
        similar_users = user_similarity_df[user].drop(user).sort_values(ascending=False)
        
        # Take top k similar users with positive similarity
        top_similar = similar_users[similar_users > 0].head(k)
        
        if len(top_similar) == 0:
            # No similar users, use user's mean rating
            predictions.loc[user] = user_ratings_mean[user]
            continue
        
        # For each item, predict based on similar users' ratings
        for item in rating_matrix.columns:
            # Get ratings from similar users for this item
            similar_users_ratings = rating_matrix.loc[top_similar.index, item]
            
            # Only use users who rated this item
            rated_users = similar_users_ratings[similar_users_ratings > 0]
            
            if len(rated_users) == 0:
                # No similar user rated this item, use user mean
                predictions.loc[user, item] = user_ratings_mean[user]
            else:
                # Weighted average
                similarities = top_similar[rated_users.index]
                weighted_sum = (rated_users * similarities).sum()
                sim_sum = similarities.sum()
                predictions.loc[user, item] = weighted_sum / sim_sum if sim_sum > 0 else user_ratings_mean[user]
    
    return predictions.astype(float)

cf_predictions = collaborative_filtering_predict(rating_matrix, k=5)
print(f"\n🤖 CF Predictions generated:")
print(f"  - Min: {cf_predictions.min().min():.2f}")
print(f"  - Max: {cf_predictions.max().max():.2f}")
print(f"  - Mean: {cf_predictions.mean().mean():.2f}")
print(f"  - Std: {cf_predictions.std().std():.2f}")

# ---------------------
# 9️⃣ Content-Based Filtering
# ---------------------
# Prepare items dataframe
movies_data = content_df[['movieid', 'title', 'genres', 'cast', 'director', 'description', 'keywords']].dropna(subset=['movieid']).rename(columns={'movieid':'id'})
series_data = content_df[['seriesid', 'title', 'genres', 'cast', 'director', 'description', 'keywords']].dropna(subset=['seriesid']).rename(columns={'seriesid':'id'})

# 🔥 FIX: Convert UUID to string
movies_data['id'] = movies_data['id'].astype(str)
series_data['id'] = series_data['id'].astype(str)

items_df = pd.concat([movies_data, series_data], ignore_index=True)

# 🔥 FIX: Only keep items that exist in rating matrix
items_in_ratings = set(rating_matrix.columns)
print(f"\n📚 Items for content-based:")
print(f"  - Total items in content_df: {len(items_df)}")
print(f"  - Items in rating matrix: {len(items_in_ratings)}")

# Debug: Show actual IDs
print(f"\n🔍 DEBUG - Sample IDs:")
print(f"  - Items from content_df movieid (first 5): {list(items_df['id'].head())}")
print(f"  - Items from rating matrix (first 5): {list(rating_matrix.columns[:5])}")

# Check overlap directly with movieid and seriesid
items_in_ratings_list = list(items_in_ratings)
movies_overlap = set(movies_data['id']) & items_in_ratings
series_overlap = set(series_data['id']) & items_in_ratings

print(f"  - Movies overlap: {len(movies_overlap)}")
print(f"  - Series overlap: {len(series_overlap)}")
print(f"  - Total overlap: {len(movies_overlap) + len(series_overlap)}")

# Debug: Find missing items - REMOVED contentId logic
missing_items = items_in_ratings - movies_overlap - series_overlap
print(f"\n🔍 Missing items analysis:")
print(f"  - Missing items count: {len(missing_items)}")

if len(missing_items) > 0:
    print(f"  - Sample missing IDs: {list(missing_items)[:5]}")
    print(f"  ⚠️ These items exist in rating matrix but not in content_df!")
    print(f"  - Possible reasons: data sync issue, deleted content, or orphaned records")
    
    # Skip missing items - only use what we have
    items_df = pd.concat([
        movies_data[movies_data['id'].isin(items_in_ratings)],
        series_data[series_data['id'].isin(items_in_ratings)]
    ], ignore_index=True)
else:
    items_df = pd.concat([
        movies_data[movies_data['id'].isin(items_in_ratings)],
        series_data[series_data['id'].isin(items_in_ratings)]
    ], ignore_index=True)

print(f"\n  - ✅ Final items: {len(items_df)} items")
print(f"  - Rating matrix has: {len(items_in_ratings)} items")

if len(items_df) < len(items_in_ratings):
    print(f"  ⚠️ WARNING: {len(items_in_ratings) - len(items_df)} items in rating matrix don't have metadata")
    print(f"  - These items will be excluded from recommendations")

if len(items_df) == 0:
    print("\n❌ No items after filtering!")
    exit()

# Create content vector
items_df['content_text'] = (
    items_df['genres'].fillna('') + ' ' + items_df['genres'].fillna('') + ' ' +
    items_df['director'].fillna('') + ' ' +
    items_df['cast'].fillna('') + ' ' +
    items_df['keywords'].fillna('') + ' ' +
    items_df['description'].fillna('')
)

# Check if content_text is empty
empty_content = (items_df['content_text'].str.strip() == '').sum()
print(f"  - Items with empty content: {empty_content}")

if len(items_df) == 0:
    print("\n❌ No items to create recommendations!")
    exit()

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=1000, min_df=1)
try:
    tfidf_matrix = tfidf.fit_transform(items_df['content_text'])
    print(f"  - TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"  - Cosine similarity matrix shape: {cosine_sim.shape}")
except ValueError as e:
    print(f"\n⚠️ TF-IDF failed: {e}")
    print("Falling back to simple overlap-based similarity...")
    
    # Fallback: simple genre-based similarity
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    items_df['genres_list'] = items_df['genres'].fillna('').str.split(', ')
    genre_matrix = mlb.fit_transform(items_df['genres_list'])
    cosine_sim = cosine_similarity(genre_matrix)
    print(f"  - Fallback genre similarity shape: {cosine_sim.shape}")

# Mappings
item_idx_map = {item_id: idx for idx, item_id in enumerate(items_df['id'])}
idx_item_map = {idx: item_id for item_id, idx in item_idx_map.items()}

def predict_content_based(user_id, candidate_item_id):
    user_interactions = user_item_ratings[user_item_ratings['userId'] == user_id]
    if user_interactions.empty:
        return 0
    
    target_idx = item_idx_map.get(candidate_item_id)
    if target_idx is None:
        return 0
    
    similarities = []
    for _, row in user_interactions.iterrows():
        seen_item = row['itemid']
        rating = row['rating']
        seen_idx = item_idx_map.get(seen_item)
        
        if seen_idx is not None and seen_idx != target_idx:
            sim = cosine_sim[target_idx][seen_idx]
            if sim > 0:  # Only positive similarities
                similarities.append((sim, rating))
    
    if not similarities:
        return 0
    
    # Weighted average
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_sims = similarities[:5]  # Top 5
    
    weighted_sum = sum(sim * rating for sim, rating in top_sims)
    weight_sum = sum(sim for sim, rating in top_sims)
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0

# ---------------------
# 🔟 Hybrid Recommendation - FIXED
# ---------------------
def hybrid_recommend(user_id, top_n=10, cf_weight=0.6, cb_weight=0.4):
    # Get seen items
    seen_items = set(user_item_ratings[user_item_ratings['userId'] == user_id]['itemid'])
    # 🔥 FIX: Only consider items in items_df (which are now in rating matrix)
    all_items = set(items_df['id'])
    candidate_items = list(all_items - seen_items)
    
    print(f"\n👤 User {user_id}:")
    print(f"  - Seen items: {list(seen_items)[:3]}... ({len(seen_items)} total)")
    print(f"  - All items in system: {len(all_items)}")
    print(f"  - Candidate items: {len(candidate_items)}")
    
    if not candidate_items:
        print("  ⚠️ No candidates to recommend!")
        return pd.DataFrame()
    
    recommendations = []
    for item_id in candidate_items:
        # CF score
        if user_id in cf_predictions.index and item_id in cf_predictions.columns:
            cf_score = float(cf_predictions.loc[user_id, item_id])
            # Normalize CF score to 0-5 range
            cf_score = max(0, min(5, cf_score))
        else:
            cf_score = 0
        
        # CB score
        cb_score = predict_content_based(user_id, item_id)
        
        # Hybrid
        hybrid_score = cf_weight * cf_score + cb_weight * cb_score
        
        recommendations.append({
            'itemid': item_id,
            'cf_score': cf_score,
            'cb_score': cb_score,
            'hybrid_score': hybrid_score
        })
    
    print(f"  - Total recommendations scored: {len(recommendations)}")
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        print(f"  - CF scores - Min: {rec_df['cf_score'].min():.2f}, Max: {rec_df['cf_score'].max():.2f}, Mean: {rec_df['cf_score'].mean():.2f}")
        print(f"  - CB scores - Min: {rec_df['cb_score'].min():.2f}, Max: {rec_df['cb_score'].max():.2f}, Mean: {rec_df['cb_score'].mean():.2f}")
    
    # Sort
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values('hybrid_score', ascending=False).head(top_n)
    
    # Merge info
    recommendations_df = recommendations_df.merge(
        items_df[['id', 'title', 'genres']], 
        left_on='itemid', 
        right_on='id', 
        how='left'
    )
    
    return recommendations_df

# ---------------------
# 1️⃣1️⃣ Test
# ---------------------
if not user_item_ratings.empty:
    test_user = user_item_ratings['userId'].iloc[0]
    print(f"\n{'='*60}")
    print(f"🎯 Recommendations for user: {test_user}")
    print(f"{'='*60}")
    
    recommendations = hybrid_recommend(test_user, top_n=10)
    
    if not recommendations.empty:
        print("\n📋 Top Recommendations:")
        print(recommendations[['title', 'genres', 'cf_score', 'cb_score', 'hybrid_score']].to_string(index=False))
    else:
        print("❌ No recommendations generated")

print("\n✅ Process complete!")