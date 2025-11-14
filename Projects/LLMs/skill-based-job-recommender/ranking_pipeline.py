import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from skill_extractor import SkillExtractor

class RankingPipeline:
    def __init__(self, model_dir: str = 'models'):
        """Initialize ranking pipeline."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.skill_extractor = SkillExtractor()
        self.bm25_model = None
        self.semantic_model = None
        self.ltr_model = None
        self.job_embeddings = None
        self.jobs_df = None
        self.tokenized_corpus = None
            
    def build_bm25_index(self, jobs_df: pd.DataFrame):
        """Build BM25 index for jobs - minimal save version."""
        print("Building BM25 index...")
        import time
        start_time = time.time()
        
        self.jobs_df = jobs_df
        
        # Get corpus
        corpus = jobs_df['searchable_text'].fillna('').tolist()
        total_docs = len(corpus)
        print(f"Tokenizing {total_docs:,} documents...")
        
        # Optimized tokenization with progress tracking
        self.tokenized_corpus = []
        chunk_size = 10000
        
        for i in range(0, total_docs, chunk_size):
            chunk = corpus[i:i+chunk_size]
            tokenized_chunk = [doc.lower().split() for doc in chunk]
            self.tokenized_corpus.extend(tokenized_chunk)
            
            progress = min(i + chunk_size, total_docs)
            pct = (progress / total_docs) * 100
            elapsed = time.time() - start_time
            print(f"  Progress: {progress:,}/{total_docs:,} ({pct:.1f}%) - {elapsed:.1f}s elapsed")
        
        print(f"✓ Tokenization complete in {time.time() - start_time:.1f}s")
        
        # Build BM25 model
        print("Building BM25 model...")
        bm25_start = time.time()
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        print(f"✓ BM25 model built in {time.time() - bm25_start:.1f}s")
        
        # Save ONLY job IDs (skip tokenized corpus - it's huge!)
        print("Saving job IDs...")
        save_start = time.time()
        
        with open(self.model_dir / 'bm25_job_ids.pkl', 'wb') as f:
            pickle.dump(jobs_df['job_id'].tolist(), f)
        
        # Save a flag that BM25 is built
        (self.model_dir / 'bm25_built.flag').touch()
        
        save_time = time.time() - save_start
        print(f"✓ Job IDs saved in {save_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"✓ BM25 index built with {len(corpus):,} documents in {total_time:.1f}s")
        print("⚠ Note: Tokenized corpus not saved (will rebuild on load)")

    def load_indices(self):
        """Load pre-built indices - rebuild BM25 from processed data."""
        print("Loading indices...")
        import time
        start_time = time.time()
        
        # Check if we need to rebuild BM25
        if (self.model_dir / 'bm25_built.flag').exists():
            print("⚠ BM25 tokenized corpus not saved, rebuilding (takes ~15s)...")
            
            # Load processed jobs
            jobs_df = pd.read_parquet('data/processed_jobs.parquet')
            corpus = jobs_df['searchable_text'].fillna('').tolist()
            
            # Quick tokenization
            self.tokenized_corpus = [doc.lower().split() for doc in corpus]
            self.bm25_model = BM25Okapi(self.tokenized_corpus)
            
            print("✓ BM25 index rebuilt")
        else:
            # Old format - try to load
            with open(self.model_dir / 'bm25_index.pkl', 'rb') as f:
                bm25_data = pickle.load(f)
                self.tokenized_corpus = bm25_data['tokenized_corpus']
                self.bm25_model = BM25Okapi(self.tokenized_corpus)
        
        # Load embeddings
        self.job_embeddings = np.load(self.model_dir / 'job_embeddings.npy')
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✓ Indices loaded in {time.time() - start_time:.1f}s")
    def build_semantic_embeddings(self, jobs_df: pd.DataFrame):
        """Build semantic embeddings using sentence transformers."""
        print("Building semantic embeddings...")
        import torch
    
        # Check GPU availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model with GPU
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Create text for embedding (shorter than full searchable text)
        embedding_texts = jobs_df.apply(
            lambda row: f"{row['title']}. {row.get('skills_desc', '')}",
            axis=1
        ).fillna('').tolist()
        
        # Generate embeddings in batches
        batch_size = 64  
        embeddings = []
        for i in range(0, len(embedding_texts), batch_size):
            batch = embedding_texts[i:i+batch_size]
            batch_embeddings = self.semantic_model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        self.job_embeddings = np.vstack(embeddings)
        
        # Save embeddings
        np.save(self.model_dir / 'job_embeddings.npy', self.job_embeddings)
        print(f"Generated embeddings for {len(self.job_embeddings)} jobs")
    
    
    def search_bm25(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:
        """Stage 1: BM25 retrieval."""
        tokenized_query = query.lower().split()
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return top_indices.tolist(), top_scores.tolist()
    
    def rerank_semantic(self, query: str, candidate_indices: List[int], 
                       top_k: int = 50) -> Tuple[List[int], List[float]]:
        """Stage 2: Semantic reranking."""
        # Encode query
        query_embedding = self.semantic_model.encode([query])
        
        # Get candidate embeddings
        candidate_embeddings = self.job_embeddings[candidate_indices]
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top k
        top_k = min(top_k, len(similarities))
        top_local_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Map back to original indices
        top_indices = [candidate_indices[i] for i in top_local_indices]
        top_scores = similarities[top_local_indices].tolist()
        
        return top_indices, top_scores
    
    def compute_features(self, user_profile: Dict, job_row: pd.Series, 
                        bm25_score: float, semantic_score: float) -> Dict[str, float]:
        """Compute features for learning-to-rank."""
        features = {}
        
        # User skills
        user_skills = set(user_profile.get('skills', []))
        
        # Job skills
        job_skills_str = job_row.get('extracted_skills', '')
        job_skills = set(job_skills_str.split(',')) if job_skills_str else set()
        
        # Skill match features
        if len(job_skills) > 0:
            skill_overlap = len(user_skills & job_skills)
            features['skill_jaccard'] = skill_overlap / len(user_skills | job_skills) if (user_skills | job_skills) else 0
            features['skill_overlap_pct'] = skill_overlap / len(job_skills)
            features['num_matching_skills'] = skill_overlap
            features['num_required_skills'] = len(job_skills)
            features['skill_coverage'] = skill_overlap / len(user_skills) if user_skills else 0
        else:
            features['skill_jaccard'] = 0
            features['skill_overlap_pct'] = 0
            features['num_matching_skills'] = 0
            features['num_required_skills'] = 0
            features['skill_coverage'] = 0
        
        # Experience level match
        user_exp = user_profile.get('experience_level', 2)
        job_exp = job_row.get('experience_level_normalized', 2)
        features['experience_match'] = 1.0 - abs(user_exp - job_exp) / 4.0
        features['experience_diff'] = abs(user_exp - job_exp)
        
        # Location match
        user_city = user_profile.get('city', '').lower()
        user_state = user_profile.get('state', '').lower()
        job_city = str(job_row.get('city_parsed', '')).lower()
        job_state = str(job_row.get('state_parsed', '')).lower()
        
        features['city_match'] = 1.0 if user_city and user_city == job_city else 0.0
        features['state_match'] = 1.0 if user_state and user_state == job_state else 0.0
        features['remote_available'] = float(job_row.get('is_remote', 0))
        features['location_score'] = features['city_match'] * 1.0 + features['state_match'] * 0.5 + features['remote_available'] * 0.3
        
        # Salary match
        user_min_salary = user_profile.get('min_salary', 0)
        job_salary = job_row.get('salary_yearly_usd', 0)
        if user_min_salary > 0 and job_salary > 0:
            features['salary_match'] = 1.0 if job_salary >= user_min_salary else job_salary / user_min_salary
        else:
            features['salary_match'] = 0.5  # Unknown
        features['salary_value'] = job_salary / 100000 if job_salary > 0 else 0  # Normalized
        
        # Company size preference
        user_company_size = user_profile.get('company_size', -1)
        job_company_size = job_row.get('company_size', -1)
        if user_company_size >= 0 and job_company_size >= 0:
            features['company_size_match'] = 1.0 - abs(user_company_size - job_company_size) / 7.0
        else:
            features['company_size_match'] = 0.5
        
        # Benefits match
        user_benefits = set(user_profile.get('benefits', []))
        job_benefits_str = job_row.get('benefits_str', '')
        job_benefits = set(job_benefits_str.split(',')) if job_benefits_str else set()
        if user_benefits and job_benefits:
            features['benefits_match'] = len(user_benefits & job_benefits) / len(user_benefits)
        else:
            features['benefits_match'] = 0.0
        
        # Popularity signals
        features['log_applies'] = np.log1p(job_row.get('applies', 0))
        features['log_views'] = np.log1p(job_row.get('views', 0))
        features['applies_normalized'] = min(job_row.get('applies', 0) / 1000, 1.0)
        
        # Ranking scores
        features['bm25_score'] = bm25_score
        features['semantic_score'] = semantic_score
        features['combined_score'] = 0.4 * bm25_score + 0.6 * semantic_score
        
        return features
    
    def rank_jobs(self, user_profile: Dict, top_k: int = 50) -> pd.DataFrame:
        """Full ranking pipeline."""
        # Build query from user profile
        query_parts = []
        
        # Skills
        skills = user_profile.get('skills', [])
        if skills:
            # Expand query with related skills
            expanded_skills = self.skill_extractor.expand_query_skills(set(skills))
            query_parts.append(' '.join(expanded_skills))
        
        # Experience level
        exp_level = user_profile.get('experience_level', 2)
        exp_map = {0: 'entry', 1: 'associate', 2: 'mid', 3: 'senior', 4: 'director'}
        query_parts.append(exp_map.get(exp_level, 'mid'))
        
        # Location
        if user_profile.get('city'):
            query_parts.append(user_profile['city'])
        if user_profile.get('remote_only'):
            query_parts.append('remote')
        
        query = ' '.join(query_parts)
        
        # Stage 1: BM25
        bm25_indices, bm25_scores = self.search_bm25(query, top_k=100)
        
        # Stage 2: Semantic reranking
        semantic_indices, semantic_scores = self.rerank_semantic(query, bm25_indices, top_k=top_k)
        
        # Get corresponding BM25 scores for semantic results
        bm25_score_dict = dict(zip(bm25_indices, bm25_scores))
        final_bm25_scores = [bm25_score_dict[idx] for idx in semantic_indices]
        
        # Prepare results
        results = []
        for idx, bm25_score, semantic_score in zip(semantic_indices, final_bm25_scores, semantic_scores):
            job_row = self.jobs_df.iloc[idx]
            
            # Compute features
            features = self.compute_features(user_profile, job_row, bm25_score, semantic_score)
            
            # Stage 3: LTR scoring (if model is loaded)
            if self.ltr_model is not None:
                feature_vector = [features[k] for k in sorted(features.keys())]
                ltr_score = self.ltr_model.predict([feature_vector])[0]
            else:
                # Fallback: weighted combination
                ltr_score = (
                    features['skill_jaccard'] * 0.3 +
                    features['semantic_score'] * 0.25 +
                    features['experience_match'] * 0.15 +
                    features['location_score'] * 0.1 +
                    features['salary_match'] * 0.1 +
                    features['bm25_score'] * 0.1
                )
            
            result = {
                'job_id': job_row['job_id'],
                'score': ltr_score,
                'bm25_score': bm25_score,
                'semantic_score': semantic_score,
                **features
            }
            results.append(result)
        
        # Sort by final score
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        # Merge with job data
        results_df = results_df.merge(self.jobs_df, on='job_id', how='left')
        
        return results_df
    
    def load_ltr_model(self, model_path: str = None):
        """Load trained LTR model."""
        if model_path is None:
            model_path = self.model_dir / 'ltr_model.pkl'
        
        if Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.ltr_model = pickle.load(f)
            print("LTR model loaded")
        else:
            print("No LTR model found, using fallback scoring")
    
    def analyze_match(self, user_profile: Dict, job_row: pd.Series) -> Dict:
        """Detailed match analysis for a single job."""
        user_skills = set(user_profile.get('skills', []))
        job_skills_str = job_row.get('extracted_skills', '')
        job_skills = set(job_skills_str.split(',')) if job_skills_str else set()
        
        matching_skills = user_skills & job_skills
        missing_skills = job_skills - user_skills
        
        analysis = {
            'matching_skills': sorted(list(matching_skills)),
            'missing_skills': sorted(list(missing_skills)),
            'match_percentage': len(matching_skills) / len(job_skills) * 100 if job_skills else 0,
            'total_required': len(job_skills),
            'total_matched': len(matching_skills)
        }
        
        return analysis