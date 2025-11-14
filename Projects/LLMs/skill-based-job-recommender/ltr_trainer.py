import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import pickle
from pathlib import Path
from ranking_pipeline import RankingPipeline

class LTRTrainer:
    def __init__(self, ranking_pipeline: RankingPipeline, model_dir: str = 'models'):
        """Initialize LTR trainer."""
        self.ranking_pipeline = ranking_pipeline
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = None
        
    def generate_synthetic_training_data(self, jobs_df: pd.DataFrame, 
                                        num_queries: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data with relevance labels."""
        print(f"Generating {num_queries} synthetic training queries...")
        
        all_features = []
        all_labels = []
        all_qids = []
        
        # Get all unique skills from dataset
        all_skills = set()
        for skills_str in jobs_df['extracted_skills'].dropna():
            if skills_str:
                all_skills.update(skills_str.split(','))
        all_skills = list(all_skills)
        
        for qid in range(num_queries):
            # Generate synthetic user profile
            num_skills = np.random.randint(3, 10)
            user_skills = list(np.random.choice(all_skills, size=min(num_skills, len(all_skills)), replace=False))
            
            user_profile = {
                'skills': user_skills,
                'experience_level': np.random.randint(0, 5),
                'min_salary': np.random.choice([0, 50000, 70000, 90000, 120000]),
                'city': np.random.choice(['', 'New York', 'San Francisco', 'Seattle', 'Austin']),
                'state': np.random.choice(['', 'NY', 'CA', 'WA', 'TX']),
                'remote_only': np.random.choice([True, False]),
                'company_size': np.random.randint(-1, 8),
                'benefits': []
            }
            
            # Sample candidate jobs
            num_candidates = min(50, len(jobs_df))
            candidate_jobs = jobs_df.sample(n=num_candidates)
            
            for _, job_row in candidate_jobs.iterrows():
                # Compute features
                features = self.ranking_pipeline.compute_features(
                    user_profile, 
                    job_row,
                    bm25_score=np.random.random(),
                    semantic_score=np.random.random()
                )
                
                # Assign relevance label based on features
                label = self._compute_relevance_label(features)
                
                # Store feature names
                if self.feature_names is None:
                    self.feature_names = sorted(features.keys())
                
                # Create feature vector
                feature_vector = [features[k] for k in self.feature_names]
                
                all_features.append(feature_vector)
                all_labels.append(label)
                all_qids.append(qid)
        
        return np.array(all_features), np.array(all_labels), np.array(all_qids)
    
    def _compute_relevance_label(self, features: Dict[str, float]) -> int:
        """Compute relevance label (0-4) based on features."""
        score = 0.0
        
        # Skill match is most important
        score += features['skill_jaccard'] * 40
        score += features['skill_overlap_pct'] * 30
        
        # Experience match
        score += features['experience_match'] * 10
        
        # Location
        score += features['location_score'] * 10
        
        # Salary
        score += features['salary_match'] * 10
        
        # Convert to 0-4 scale
        if score >= 80:
            return 4  # Highly relevant
        elif score >= 60:
            return 3  # Relevant
        elif score >= 40:
            return 2  # Somewhat relevant
        elif score >= 20:
            return 1  # Marginally relevant
        else:
            return 0  # Not relevant
    
    def train_model(self, X: np.ndarray, y: np.ndarray, qids: np.ndarray):
        """Train LightGBM LambdaRank model."""
        print("Training LTR model...")
        
        # Split data
        unique_qids = np.unique(qids)
        train_qids, val_qids = train_test_split(unique_qids, test_size=0.2, random_state=42)
        
        train_mask = np.isin(qids, train_qids)
        val_mask = np.isin(qids, val_qids)
        
        X_train, y_train, qids_train = X[train_mask], y[train_mask], qids[train_mask]
        X_val, y_val, qids_val = X[val_mask], y[val_mask], qids[val_mask]
        
        # Create group information (number of instances per query)
        train_groups = [np.sum(qids_train == qid) for qid in np.unique(qids_train)]
        val_groups = [np.sum(qids_val == qid) for qid in np.unique(qids_val)]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=self.feature_names, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5, 10],
            'learning_rate': 0.05,
             'device': 'gpu',  # Add this
              'gpu_platform_id': 0,  # Add this
              'gpu_device_id': 0,  # Add this
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20)]
        )
        
        print("Training completed")
        
        # Save model
        with open(self.model_dir / 'ltr_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names
        with open(self.model_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"Model saved to {self.model_dir}")
        
        return model
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, qids: np.ndarray):
        """Evaluate model performance."""
        from sklearn.metrics import ndcg_score
        
        print("\nModel Evaluation:")
        
        # Predict
        y_pred = model.predict(X)
        
        # Calculate NDCG for each query
        ndcg_scores = []
        for qid in np.unique(qids):
            mask = qids == qid
            y_true_q = y[mask].reshape(1, -1)
            y_pred_q = y_pred[mask].reshape(1, -1)
            
            if len(y_true_q[0]) > 1:  # Need at least 2 items
                ndcg = ndcg_score(y_true_q, y_pred_q)
                ndcg_scores.append(ndcg)
        
        print(f"Average NDCG: {np.mean(ndcg_scores):.4f}")
        print(f"Median NDCG: {np.median(ndcg_scores):.4f}")
        
        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'mean_ndcg': np.mean(ndcg_scores),
            'median_ndcg': np.median(ndcg_scores),
            'feature_importance': feature_importance
        }
    
    def compare_ranking_stages(self, jobs_df: pd.DataFrame, num_test_queries: int = 100):
        """Compare ranking quality at each stage."""
        print(f"\nComparing ranking stages with {num_test_queries} test queries...")
        
        # Get all skills
        all_skills = set()
        for skills_str in jobs_df['extracted_skills'].dropna():
            if skills_str:
                all_skills.update(skills_str.split(','))
        all_skills = list(all_skills)
        
        results = {
            'bm25_only': [],
            'bm25_semantic': [],
            'full_pipeline': []
        }
        
        for _ in range(num_test_queries):
            # Generate test query
            num_skills = np.random.randint(3, 8)
            user_skills = list(np.random.choice(all_skills, size=min(num_skills, len(all_skills)), replace=False))
            
            user_profile = {
                'skills': user_skills,
                'experience_level': np.random.randint(0, 5),
                'min_salary': 70000,
                'city': 'San Francisco',
                'state': 'CA',
                'remote_only': False,
                'company_size': 3,
                'benefits': []
            }
            
            # Get rankings at each stage
            query = ' '.join(user_skills)
            
            # Stage 1: BM25 only
            bm25_indices, bm25_scores = self.ranking_pipeline.search_bm25(query, top_k=10)
            bm25_avg_score = np.mean(bm25_scores)
            
            # Stage 2: BM25 + Semantic
            semantic_indices, semantic_scores = self.ranking_pipeline.rerank_semantic(query, bm25_indices[:50], top_k=10)
            semantic_avg_score = np.mean(semantic_scores)
            
            # Stage 3: Full pipeline with LTR
            full_results = self.ranking_pipeline.rank_jobs(user_profile, top_k=10)
            full_avg_score = full_results['score'].mean()
            
            results['bm25_only'].append(bm25_avg_score)
            results['bm25_semantic'].append(semantic_avg_score)
            results['full_pipeline'].append(full_avg_score)
        
        print("\nRanking Stage Comparison:")
        print(f"BM25 Only - Mean Score: {np.mean(results['bm25_only']):.4f}")
        print(f"BM25 + Semantic - Mean Score: {np.mean(results['bm25_semantic']):.4f}")
        print(f"Full Pipeline (LTR) - Mean Score: {np.mean(results['full_pipeline']):.4f}")
        
        return results