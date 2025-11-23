from sklearn.feature_extraction.text import TfidfVectorizer
from config.setting import TFIDF_MAX_FEATURES
from utils.logger import log

def build_tfidf_matrix(items_df):
    items_df["content_text"] = (
        items_df["genres"].fillna("") + " " +
        items_df["director"].fillna("") + " " +
        items_df["cast"].fillna("") + " " +
        items_df["keywords"].fillna("") + " " +
        items_df["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=TFIDF_MAX_FEATURES,
        min_df=1
    )

    log("Building TF-IDF matrix...")
    tfidf_mx = vectorizer.fit_transform(items_df["content_text"])

    return tfidf_mx
