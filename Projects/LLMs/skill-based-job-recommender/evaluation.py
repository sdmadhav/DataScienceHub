"""
Evaluation script for the job matching system.
Analyzes performance metrics and feature importance.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

from ranking_pipeline import RankingPipeline
from ltr_trainer import LTRTrainer

def evaluate_ndcg():
    """Calculate NDCG metrics for the system."""
    print("=" * 60)
    print("NDCG Evaluation")
    print("=" * 60)
    
    # Load data and models
    jobs_df = pd.read_parquet('data/processed_jobs.parquet')
    pipeline = RankingPipeline(model_dir='models')
    pipeline.jobs_df = jobs_df
    pipeline.load_indices()
    pipeline.load_ltr_model()
    
    trainer = LTRTrainer(pipeline)
    
    # Generate test data
    print("Generating test queries...")
    X_test, y_test, qids_test = trainer.generate_synthetic_training_data(jobs_df, num_queries=200)
    
    # Evaluate
    print("\nEvaluating model...")
    
    # Load model
    with open('models/ltr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate NDCG for different k values
    ndcg_at_k = {}
    for k in [1, 3, 5, 10, 20]:
        ndcg_scores = []
        for qid in np.unique(qids_test):
            mask = qids_test == qid
            y_true_q = y_test[mask]
            y_pred_q = y_pred[mask]
            
            if len(y_true_q) >= k:
                # Get top k
                top_k_indices = np.argsort(y_pred_q)[::-1][:k]
                y_true_k = y_true_q[top_k_indices].reshape(1, -1)
                y_pred_k = y_pred_q[top_k_indices].reshape(1, -1)
                
                ndcg = ndcg_score(y_true_k, y_pred_k)
                ndcg_scores.append(ndcg)
        
        ndcg_at_k[k] = np.mean(ndcg_scores)
    
    print("\nNDCG Scores:")
    for k, score in ndcg_at_k.items():
        print(f"  NDCG@{k}: {score:.4f}")
    
    return ndcg_at_k

def analyze_feature_importance():
    """Analyze and visualize feature importance."""
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    # Load model
    with open('models/ltr_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Save to CSV
    feature_importance.to_csv('evaluation_results/feature_importance.csv', index=False)
    print("\n✓ Saved to 'evaluation_results/feature_importance.csv'")
    
    return feature_importance

def compare_ranking_stages():
    """Compare performance at each ranking stage."""
    print("\n" + "=" * 60)
    print("Ranking Stage Comparison")
    print("=" * 60)
    
    jobs_df = pd.read_parquet('data/processed_jobs.parquet')
    pipeline = RankingPipeline(model_dir='models')
    pipeline.jobs_df = jobs_df
    pipeline.load_indices()
    pipeline.load_ltr_model()
    
    trainer = LTRTrainer(pipeline)
    results = trainer.compare_ranking_stages(jobs_df, num_test_queries=100)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results/stage_comparison.csv', index=False)
    print("\n✓ Saved to 'evaluation_results/stage_comparison.csv'")
    
    return results

def analyze_example_queries():
    """Run example queries and analyze results."""
    print("\n" + "=" * 60)
    print("Example Query Analysis")
    print("=" * 60)
    
    jobs_df = pd.read_parquet('data/processed_jobs.parquet')
    pipeline = RankingPipeline(model_dir='models')
    pipeline.jobs_df = jobs_df
    pipeline.load_indices()
    pipeline.load_ltr_model()
    
    # Example user profiles
    examples = [
        {
            'name': 'Python Data Scientist',
            'profile': {
                'skills': ['python', 'machine learning', 'pandas', 'numpy', 'sql'],
                'experience_level': 2,
                'min_salary': 100000,
                'city': 'San Francisco',
                'state': 'CA',
                'remote_only': False,
                'company_size': 3,
                'benefits': []
            }
        },
        {
            'name': 'Frontend Developer',
            'profile': {
                'skills': ['react', 'javascript', 'typescript', 'html', 'css'],
                'experience_level': 1,
                'min_salary': 80000,
                'city': 'New York',
                'state': 'NY',
                'remote_only': True,
                'company_size': -1,
                'benefits': []
            }
        },
        {
            'name': 'DevOps Engineer',
            'profile': {
                'skills': ['aws', 'docker', 'kubernetes', 'python', 'terraform'],
                'experience_level': 3,
                'min_salary': 120000,
                'city': 'Seattle',
                'state': 'WA',
                'remote_only': False,
                'company_size': 5,
                'benefits': []
            }
        }
    ]
    
    all_results = []
    
    for example in examples:
        print(f"\n{'='*40}")
        print(f"Query: {example['name']}")
        print(f"Skills: {', '.join(example['profile']['skills'])}")
        print(f"{'='*40}")
        
        results = pipeline.rank_jobs(example['profile'], top_k=10)
        
        print(f"\nTop 5 Results:")
        for i, (_, row) in enumerate(results.head(5).iterrows(), 1):
            print(f"\n{i}. {row['title']}")
            print(f"   Company: {row.get('company_name', 'Unknown')}")
            print(f"   Match Score: {row['score']:.3f}")
            print(f"   Skill Match: {row['skill_jaccard']*100:.1f}%")
            
            # Skill analysis
            analysis = pipeline.analyze_match(example['profile'], row)
            print(f"   Matching Skills ({len(analysis['matching_skills'])}): {', '.join(analysis['matching_skills'][:5])}")
            if analysis['missing_skills']:
                print(f"   Missing Skills ({len(analysis['missing_skills'])}): {', '.join(analysis['missing_skills'][:3])}")
        
        # Store results
        for _, row in results.head(5).iterrows():
            all_results.append({
                'query': example['name'],
                'job_title': row['title'],
                'company': row.get('company_name', 'Unknown'),
                'score': row['score'],
                'skill_match': row['skill_jaccard'] * 100
            })
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('evaluation_results/example_queries.csv', index=False)
    print(f"\n\n✓ Saved to 'evaluation_results/example_queries.csv'")
    
    return results_df

def main():
    """Run all evaluations."""
    # Create output directory
    Path('evaluation_results').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Job Matching System - Evaluation Suite")
    print("=" * 60)
    
    try:
        # 1. NDCG Evaluation
        ndcg_results = evaluate_ndcg()
        
        # 2. Feature Importance
        feature_importance = analyze_feature_importance()
        
        # 3. Stage Comparison
        stage_results = compare_ranking_stages()
        
        # 4. Example Queries
        example_results = analyze_example_queries()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ Evaluation Complete!")
        print("=" * 60)
        print("\nGenerated Reports:")
        print("  ✓ evaluation_results/feature_importance.csv")
        print("  ✓ evaluation_results/stage_comparison.csv")
        print("  ✓ evaluation_results/example_queries.csv")
        
        print("\n📊 Summary:")
        print(f"  Average NDCG@10: {ndcg_results[10]:.4f}")
        print(f"  Top Feature: {feature_importance.iloc[0]['feature']}")
        print("  All evaluation results saved to 'evaluation_results/' directory")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()