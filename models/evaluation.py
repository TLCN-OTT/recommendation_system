import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.logger import log
import time
from datetime import datetime

def evaluate_model(user_item_df, cf_predictions, test_size=0.2, model_name="Model"):
    """
    Evaluate recommendation model using train/test split with detailed metrics
    
    Args:
        user_item_df: DataFrame with user-item interactions
        cf_predictions: Prediction matrix
        test_size: Proportion of test data
        model_name: Name of the algorithm being evaluated
    """
    start_time = time.time()
    
    # Split data
    train_df, test_df = train_test_split(
        user_item_df, 
        test_size=test_size, 
        random_state=42,
        stratify=user_item_df['userId']  # Ensure each user in both sets
    )
    
    # Log dataset statistics
    log(f"\n{'='*60}")
    log(f"Evaluating {model_name}")
    log(f"{'='*60}")
    log(f"Dataset Split:")
    log(f"  Total interactions: {len(user_item_df)}")
    log(f"  Training samples: {len(train_df)} ({(1-test_size)*100:.1f}%)")
    log(f"  Test samples: {len(test_df)} ({test_size*100:.1f}%)")
    log(f"  Unique users (train): {train_df['userId'].nunique()}")
    log(f"  Unique items (train): {train_df['itemid'].nunique()}")
    log(f"  Unique users (test): {test_df['userId'].nunique()}")
    log(f"  Unique items (test): {test_df['itemid'].nunique()}")
    
    # Prepare test predictions
    y_true = []
    y_pred = []
    missing_predictions = 0
    
    for _, row in test_df.iterrows():
        user_id = row['userId']
        item_id = row['itemid']
        true_rating = row['rating']
        
        if user_id in cf_predictions.index and item_id in cf_predictions.columns:
            pred_rating = cf_predictions.loc[user_id, item_id]
            if not np.isnan(pred_rating):
                y_true.append(true_rating)
                y_pred.append(pred_rating)
            else:
                missing_predictions += 1
        else:
            missing_predictions += 1
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Additional metrics
    coverage = (len(y_pred) / len(test_df)) * 100
    
    # Prediction distribution analysis
    y_pred_array = np.array(y_pred)
    y_true_array = np.array(y_true)
    
    execution_time = time.time() - start_time
    
    log(f"\nPrediction Coverage:")
    log(f"  Successful predictions: {len(y_pred)}")
    log(f"  Missing predictions: {missing_predictions}")
    log(f"  Coverage rate: {coverage:.2f}%")
    
    log(f"\nEvaluation Metrics:")
    log(f"  RMSE: {rmse:.4f}")
    log(f"  MAE: {mae:.4f}")
    
    log(f"\nPrediction Statistics:")
    log(f"  Mean predicted rating: {y_pred_array.mean():.4f}")
    log(f"  Mean actual rating: {y_true_array.mean():.4f}")
    log(f"  Std predicted rating: {y_pred_array.std():.4f}")
    log(f"  Std actual rating: {y_true_array.std():.4f}")
    log(f"  Min predicted rating: {y_pred_array.min():.4f}")
    log(f"  Max predicted rating: {y_pred_array.max():.4f}")
    
    log(f"\nExecution Time: {execution_time:.2f} seconds")
    log(f"{'='*60}\n")
    
    return {
        'model_name': model_name,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(y_true),
        'coverage': coverage,
        'missing_predictions': missing_predictions,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'mean_pred': y_pred_array.mean(),
        'mean_true': y_true_array.mean(),
        'execution_time': execution_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def evaluate_ranking(user_id, recommendations, test_interactions, k=10):
    """
    Evaluate ranking quality using Precision@K and Recall@K
    """
    # Get actual test items for user
    actual_items = set(test_interactions[
        test_interactions['userId'] == user_id
    ]['itemid'].values)
    
    if not actual_items:
        return None
    
    # Get recommended items
    recommended_items = set(recommendations.head(k)['itemid'].values)
    
    # Calculate metrics
    hits = len(actual_items & recommended_items)
    precision = hits / k if k > 0 else 0
    recall = hits / len(actual_items) if len(actual_items) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision@k': precision,
        'recall@k': recall,
        'f1@k': f1,
        'hits': hits,
        'total_relevant': len(actual_items)
    }

def calculate_weighted_rating(interactions_df, weights=None):
    """
    Calculate weighted rating based on different interaction types
    
    Args:
        interactions_df: DataFrame with interaction columns (click, view, time_spent, etc.)
        weights: Dictionary with weights for each interaction type
                 Example: {'click': 1.0, 'view': 0.5, 'time_spent': 2.0, 'purchase': 5.0}
    
    Returns:
        DataFrame with weighted rating column
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'click': 1.0,
            'view': 0.5,
            'time_spent': 2.0,
            'purchase': 5.0,
            'rating': 3.0,
            'favorite': 4.0,
            'share': 3.5
        }
    
    log(f"\nCalculating Weighted Ratings:")
    log(f"Weights configuration:")
    for feature, weight in weights.items():
        log(f"  {feature}: {weight:.2f}")
    
    df = interactions_df.copy()
    df['weighted_rating'] = 0.0
    
    # Calculate weighted sum
    feature_contributions = {}
    for feature, weight in weights.items():
        if feature in df.columns:
            # Normalize feature values to 0-1 range if needed
            if feature == 'time_spent':
                max_time = df[feature].max()
                if max_time > 0:
                    normalized = df[feature] / max_time
                else:
                    normalized = df[feature]
            else:
                normalized = df[feature]
            
            contribution = normalized * weight
            df['weighted_rating'] += contribution
            feature_contributions[feature] = contribution.sum()
    
    # Normalize to typical rating scale (e.g., 1-5)
    max_rating = df['weighted_rating'].max()
    if max_rating > 0:
        df['weighted_rating'] = 1 + (df['weighted_rating'] / max_rating) * 4
    
    log(f"\nFeature Contributions:")
    total_contribution = sum(feature_contributions.values())
    for feature, contribution in feature_contributions.items():
        percentage = (contribution / total_contribution * 100) if total_contribution > 0 else 0
        log(f"  {feature}: {percentage:.2f}%")
    
    log(f"\nWeighted Rating Statistics:")
    log(f"  Mean: {df['weighted_rating'].mean():.4f}")
    log(f"  Median: {df['weighted_rating'].median():.4f}")
    log(f"  Std: {df['weighted_rating'].std():.4f}")
    log(f"  Min: {df['weighted_rating'].min():.4f}")
    log(f"  Max: {df['weighted_rating'].max():.4f}")
    
    return df

def evaluate_hybrid_model(user_item_df, cf_predictions, sim_mx, content, item_index, 
                          test_size=0.2, alpha=0.5, beta=0.3, gamma=0.2, model_name="Hybrid Model"):
    """
    Evaluate Hybrid recommendation model (CF + Content-Based + Popularity)
    
    Args:
        user_item_df: DataFrame with user-item interactions
        cf_predictions: Collaborative filtering predictions
        sim_mx: Content similarity matrix
        content: Content DataFrame
        item_index: Dictionary mapping item_id to index
        test_size: Test split ratio
        alpha: CF weight
        beta: Content-based weight
        gamma: Popularity weight
        model_name: Model name for logging
    """
    from models.hybrid import hybrid_recommend
    
    start_time = time.time()
    
    log(f"\n{'='*60}")
    log(f"Evaluating {model_name}")
    log(f"{'='*60}")
    log(f"Hybrid Weights:")
    log(f"  CF Weight (alpha): {alpha:.2f}")
    log(f"  Content-Based Weight (beta): {beta:.2f}")
    log(f"  Popularity Weight (gamma): {gamma:.2f}")
    
    # Split data
    train_df, test_df = train_test_split(
        user_item_df, 
        test_size=test_size, 
        random_state=42,
        stratify=user_item_df['userId']
    )
    
    log(f"\nDataset Split:")
    log(f"  Total interactions: {len(user_item_df)}")
    log(f"  Training samples: {len(train_df)} ({(1-test_size)*100:.1f}%)")
    log(f"  Test samples: {len(test_df)} ({test_size*100:.1f}%)")
    
    # Evaluate for each user - FIRST collect all scores
    all_scores = []
    test_data = []
    missing_predictions = 0
    
    test_users = test_df['userId'].unique()
    log(f"\nEvaluating {len(test_users)} users...")
    
    for idx, user_id in enumerate(test_users):
        if (idx + 1) % 10 == 0:
            log(f"  Processing user {idx + 1}/{len(test_users)}...")
        
        # Get user's test items
        user_test = test_df[test_df['userId'] == user_id]
        
        # Generate hybrid recommendations using only training data
        try:
            train_user_item = train_df.copy()
            
            recommendations = hybrid_recommend(
                user_id, 
                train_user_item,
                cf_predictions, 
                sim_mx, 
                content, 
                item_index,
                top_n=100,  # Get more recommendations for better coverage
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            
            # Get predictions for test items
            for _, row in user_test.iterrows():
                item_id = row['itemid']
                true_rating = row['rating']
                
                # Find predicted score for this item
                rec_row = recommendations[recommendations['itemid'] == item_id]
                if not rec_row.empty:
                    pred_score = rec_row.iloc[0]['hybrid_score']
                    all_scores.append(pred_score)
                    test_data.append((true_rating, pred_score))
                else:
                    missing_predictions += 1
        except Exception as e:
            missing_predictions += len(user_test)
    
    # Now normalize ALL scores to 1-5 scale
    y_true = []
    y_pred = []
    
    if len(all_scores) > 0:
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1
        
        for true_rating, pred_score in test_data:
            # Normalize to 1-5 scale
            pred_rating = 1 + ((pred_score - min_score) / score_range) * 4
            y_true.append(true_rating)
            y_pred.append(pred_rating)
    
    # Calculate metrics
    if len(y_true) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        coverage = (len(y_pred) / len(test_df)) * 100
        
        y_pred_array = np.array(y_pred)
        y_true_array = np.array(y_true)
        
        execution_time = time.time() - start_time
        
        log(f"\nPrediction Coverage:")
        log(f"  Successful predictions: {len(y_pred)}")
        log(f"  Missing predictions: {missing_predictions}")
        log(f"  Coverage rate: {coverage:.2f}%")
        
        log(f"\nEvaluation Metrics:")
        log(f"  RMSE: {rmse:.4f}")
        log(f"  MAE: {mae:.4f}")
        
        log(f"\nPrediction Statistics:")
        log(f"  Mean predicted rating: {y_pred_array.mean():.4f}")
        log(f"  Mean actual rating: {y_true_array.mean():.4f}")
        log(f"  Std predicted rating: {y_pred_array.std():.4f}")
        log(f"  Std actual rating: {y_true_array.std():.4f}")
        
        log(f"\nExecution Time: {execution_time:.2f} seconds")
        log(f"{'='*60}\n")
        
        return {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_true),
            'coverage': coverage,
            'missing_predictions': missing_predictions,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'mean_pred': y_pred_array.mean(),
            'mean_true': y_true_array.mean(),
            'execution_time': execution_time,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        log("ERROR: No predictions could be made!")
        return None

def compare_algorithms(evaluation_results):
    """
    Compare multiple algorithms side by side
    
    Args:
        evaluation_results: List of evaluation result dictionaries
    """
    log(f"\n{'='*80}")
    log("Algorithm Comparison")
    log(f"{'='*80}")
    
    # Create comparison table
    df_comparison = pd.DataFrame(evaluation_results)
    
    # Sort by RMSE (lower is better)
    df_comparison = df_comparison.sort_values('rmse')
    
    log(f"\n{'Model':<35} {'RMSE':<10} {'MAE':<10} {'Coverage':<12} {'Time(s)':<10}")
    log(f"{'-'*80}")
    
    for _, row in df_comparison.iterrows():
        log(f"{row['model_name']:<35} {row['rmse']:<10.4f} {row['mae']:<10.4f} "
            f"{row['coverage']:<11.2f}% {row['execution_time']:<10.2f}")
    
    # Find best model
    best_model = df_comparison.iloc[0]
    log(f"\n{'='*80}")
    log(f"Best Model: {best_model['model_name']}")
    log(f"  RMSE: {best_model['rmse']:.4f}")
    log(f"  MAE: {best_model['mae']:.4f}")
    log(f"  Coverage: {best_model['coverage']:.2f}%")
    
    # Show hybrid weights if available
    if 'alpha' in best_model and pd.notna(best_model['alpha']):
        log(f"  Optimal Hybrid Weights:")
        log(f"    CF (alpha): {best_model['alpha']:.2f}")
        log(f"    CB (beta): {best_model['beta']:.2f}")
        log(f"    Pop (gamma): {best_model['gamma']:.2f}")
    
    log(f"{'='*80}\n")
    
    return df_comparison

def evaluate_with_weights(user_item_df, cf_predictions, weights, test_size=0.2, model_name="Weighted Model"):
    """
    Evaluate model with custom weights applied to interactions
    
    Args:
        user_item_df: DataFrame with interaction data
        cf_predictions: Prediction matrix
        weights: Dictionary of feature weights
        test_size: Test split ratio
        model_name: Model name for logging
    """
    # Calculate weighted ratings
    df_weighted = calculate_weighted_rating(user_item_df, weights)
    
    # Use weighted ratings for evaluation
    df_weighted['rating'] = df_weighted['weighted_rating']
    
    # Evaluate using the weighted ratings
    results = evaluate_model(df_weighted, cf_predictions, test_size, model_name)
    
    return results