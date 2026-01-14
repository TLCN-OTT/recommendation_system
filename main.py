import pandas as pd
from data.load_auditlog import load_auditlog
from data.load_content import load_content
from models.cf_model import build_cf_predictions
from models.hybrid import hybrid_recommend
from features.text_features import build_tfidf_matrix
from utils.timer import Timer
from utils.logger import log
from sklearn.metrics.pairwise import cosine_similarity
from models.evaluation import (
    evaluate_model, 
    compare_algorithms, 
    calculate_weighted_rating,
    evaluate_with_weights,
    evaluate_hybrid_model
)

if __name__ == "__main__":
    try:
        # ===== LOAD DATA =====
        with Timer("Load data"):
            auditlog = load_auditlog()
            content = load_content()
            log(f"Loaded {len(auditlog)} audit log entries")
            log(f"Loaded {len(content)} content items")
        
        # ===== PREPARE RATING MATRIX =====
        with Timer("Prepare rating matrix"):
            user_item = auditlog.groupby(["userId", "itemid"])["rating"].max().reset_index()
            R = user_item.pivot_table(index="userId", columns="itemid", values="rating", fill_value=0)
            log(f"Rating matrix shape: {R.shape}")
            log(f"Sparsity: {(1 - R.astype(bool).sum().sum() / (R.shape[0] * R.shape[1])) * 100:.2f}%")
        
        # ===== COLLABORATIVE FILTERING =====
        with Timer("Collaborative Filtering"):
            cf_pred = build_cf_predictions(R)
            log(f"CF prediction matrix shape: {cf_pred.shape}")
        
        # ===== CONTENT-BASED FILTERING =====
        with Timer("Content-Based TF-IDF"):
            tfidf_mx = build_tfidf_matrix(content)
            sim_mx = cosine_similarity(tfidf_mx)
            log(f"Content similarity matrix shape: {sim_mx.shape}")
        
        item_index = {item: i for i, item in enumerate(content["itemid"])}

        # ===== MODEL EVALUATION =====
        log("\n" + "="*80)
        log("MODEL EVALUATION")
        log("="*80)
        
        evaluation_results = []
        
        # 1. Evaluate Collaborative Filtering
        with Timer("Evaluate Collaborative Filtering"):
            cf_metrics = evaluate_model(
                user_item, 
                cf_pred, 
                test_size=0.2, 
                model_name="Collaborative Filtering"
            )
            evaluation_results.append(cf_metrics)
        
        # 2. Evaluate HYBRID Model with different weight configurations
        log("\n" + "="*80)
        log("HYBRID MODEL EVALUATION - TESTING DIFFERENT WEIGHT CONFIGURATIONS")
        log("="*80)
        
        # Hybrid configuration 1: CF-focused (CF is strongest)
        with Timer("Evaluate Hybrid Model (CF-focused)"):
            hybrid_metrics_1 = evaluate_hybrid_model(
                user_item,
                cf_pred,
                sim_mx,
                content,
                item_index,
                test_size=0.2,
                alpha=0.6,  # CF weight
                beta=0.3,   # CB weight
                gamma=0.1,  # Pop weight
                model_name="Hybrid Model (CF-focused: 0.6/0.3/0.1)"
            )
            if hybrid_metrics_1:
                evaluation_results.append(hybrid_metrics_1)
        
        # Hybrid configuration 2: Balanced
        with Timer("Evaluate Hybrid Model (Balanced)"):
            hybrid_metrics_2 = evaluate_hybrid_model(
                user_item,
                cf_pred,
                sim_mx,
                content,
                item_index,
                test_size=0.2,
                alpha=0.5,  # CF weight
                beta=0.3,   # CB weight
                gamma=0.2,  # Pop weight
                model_name="Hybrid Model (Balanced: 0.5/0.3/0.2)"
            )
            if hybrid_metrics_2:
                evaluation_results.append(hybrid_metrics_2)
        
        # Hybrid configuration 3: CB-focused (Content-based is strongest)
        with Timer("Evaluate Hybrid Model (CB-focused)"):
            hybrid_metrics_3 = evaluate_hybrid_model(
                user_item,
                cf_pred,
                sim_mx,
                content,
                item_index,
                test_size=0.2,
                alpha=0.3,  # CF weight
                beta=0.5,   # CB weight
                gamma=0.2,  # Pop weight
                model_name="Hybrid Model (CB-focused: 0.3/0.5/0.2)"
            )
            if hybrid_metrics_3:
                evaluation_results.append(hybrid_metrics_3)
        
        # Hybrid configuration 4: Popularity-boosted
        with Timer("Evaluate Hybrid Model (Pop-boosted)"):
            hybrid_metrics_4 = evaluate_hybrid_model(
                user_item,
                cf_pred,
                sim_mx,
                content,
                item_index,
                test_size=0.2,
                alpha=0.4,  # CF weight
                beta=0.3,   # CB weight
                gamma=0.3,  # Pop weight
                model_name="Hybrid Model (Pop-boosted: 0.4/0.3/0.3)"
            )
            if hybrid_metrics_4:
                evaluation_results.append(hybrid_metrics_4)
        
        # 3. Evaluate with different INTERACTION weight configurations (for comparison)
        log("\n" + "="*80)
        log("INTERACTION WEIGHT TESTING (CF-based)")
        log("="*80)
        
        # Weight configuration 1: Balanced interactions
        weights_balanced = {
            'click': 1.0,
            'view': 1.0,
            'rating': 1.0
        }
        with Timer("Evaluate Weighted CF (Balanced)"):
            weighted_metrics_1 = evaluate_with_weights(
                auditlog,
                cf_pred,
                weights_balanced,
                test_size=0.2,
                model_name="Weighted CF (click:1/view:1/rating:1)"
            )
            evaluation_results.append(weighted_metrics_1)
        
        # Weight configuration 2: Rating-focused
        weights_rating_focused = {
            'click': 1.0,
            'view': 0.5,
            'rating': 4.0
        }
        with Timer("Evaluate Weighted CF (Rating-focused)"):
            weighted_metrics_2 = evaluate_with_weights(
                auditlog,
                cf_pred,
                weights_rating_focused,
                test_size=0.2,
                model_name="Weighted CF (click:1/view:0.5/rating:4)"
            )
            evaluation_results.append(weighted_metrics_2)
        
        # 3. Compare all algorithms
        with Timer("Algorithm Comparison"):
            comparison_df = compare_algorithms(evaluation_results)
            
            # Save comparison results to CSV
            output_file = "evaluation_results.csv"
            comparison_df.to_csv(output_file, index=False)
            log(f"\nEvaluation results saved to: {output_file}")
        
        # ===== DETAILED ANALYSIS =====
        log("\n" + "="*80)
        log("DETAILED ANALYSIS")
        log("="*80)
        
        # Analyze best performing model
        best_model = comparison_df.iloc[0]
        log(f"\nBest Performing Model: {best_model['model_name']}")
        log(f"Performance Metrics:")
        log(f"  RMSE: {best_model['rmse']:.4f}")
        log(f"  MAE: {best_model['mae']:.4f}")
        log(f"  Coverage: {best_model['coverage']:.2f}%")
        log(f"  Execution Time: {best_model['execution_time']:.2f}s")
        
        # Calculate improvement over baseline CF
        baseline_rmse = evaluation_results[0]['rmse']
        best_rmse = best_model['rmse']
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        log(f"\nImprovement over baseline CF:")
        log(f"  RMSE improvement: {improvement:.2f}%")
        
        # Show best hybrid configuration
        if 'alpha' in best_model:
            log(f"\nBest Hybrid Configuration:")
            log(f"  Alpha (CF): {best_model['alpha']:.2f}")
            log(f"  Beta (CB): {best_model['beta']:.2f}")
            log(f"  Gamma (Pop): {best_model['gamma']:.2f}")
        
        # ===== GENERATE RECOMMENDATIONS =====
        log("\n" + "="*80)
        log("GENERATE RECOMMENDATIONS")
        log("="*80)
        
        uid = user_item["userId"].iloc[0]
        log(f"\nGenerating recommendations for user: {uid}")
        
        with Timer("Hybrid Recommendation"):
            rec = hybrid_recommend(uid, user_item, cf_pred, sim_mx, content, item_index)
        
        log(f"\nTop 10 Recommendations:")
        print(rec[["title", "cf_score", "cb_score", "pop_score", "hybrid_score"]].head(10))
        
        # Save recommendations
        rec_output_file = f"recommendations_user_{uid}.csv"
        rec.to_csv(rec_output_file, index=False)
        log(f"\nRecommendations saved to: {rec_output_file}")
        
        # ===== SUMMARY =====
        log("\n" + "="*80)
        log("EXECUTION SUMMARY")
        log("="*80)
        log(f"Total models evaluated: {len(evaluation_results)}")
        log(f"Best model: {best_model['model_name']}")
        log(f"Best RMSE: {best_model['rmse']:.4f}")
        log(f"Best MAE: {best_model['mae']:.4f}")
        log(f"Recommendations generated for user {uid}")
        log("="*80 + "\n")
        
    except Exception as e:
        log(f"Error in main.py: {e}")
        raise