"""
Preprocessing script to load, process data, and build indices.
Run this script once before starting the application.

Usage:
    python preprocess.py             # Run full pipeline
    python preprocess.py --sample 500  # Run with a sample of 500 jobs
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from data_processor import DataProcessor
from ranking_pipeline import RankingPipeline
from ltr_trainer import LTRTrainer

def main():
    parser = argparse.ArgumentParser(description="Preprocess LinkedIn job dataset.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use only a sample of the job dataset (e.g., 1000) for testing.")
    args = parser.parse_args()

    print("=" * 60)
    print("Skills-Based Job Matching System - Data Preprocessing")
    print("=" * 60)
    
    # Check if data directory exists
    data_dir = Path('data')
    if not data_dir.exists():
        print("\n❌ Error: 'data' directory not found!")
        print("Please download the dataset from Kaggle and place it in the 'data' folder.")
        sys.exit(1)
    
    # Check for required files
    required_files = ['job_postings.csv']
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        print(f"\n❌ Error: Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("\n✓ Data directory found")
    
    # Step 1: Load and process data
    print("\n" + "=" * 60)
    print("Step 1: Loading and Processing Data")
    print("=" * 60)
    
    processor = DataProcessor(data_dir='data')
    processor.load_data()

    # ✅ Take a sample if requested
    if args.sample:
        print(f"\n🔍 Using a random sample of {args.sample} job postings for testing...\n")
        processor.jobs_df = processor.jobs_df.sample(
            n=min(args.sample, len(processor.jobs_df)), 
            random_state=42
        ).reset_index(drop=True)
    
    jobs_df = processor.process_data()
    # jobs_df = pd.read_parquet("/content/data/processed_jobs.parquet")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    stats = processor.get_skill_statistics(jobs_df)
    print(f"  Total Jobs: {stats['total_jobs']:,}")
    print(f"  Jobs with Skills: {stats['jobs_with_skills']:,}")
    print(f"  Unique Skills: {stats['unique_skills']}")
    print(f"\n  Top 10 Skills:")
    for skill, count in stats['top_skills'][:10]:
        print(f"    {skill}: {count:,}")
    
    # Save processed data
    output_path = 'data/processed_jobs_sample.parquet' if args.sample else 'data/processed_jobs.parquet'
    processor.save_processed_data(jobs_df, output_path=output_path)
    print(f"\n✓ Processed data saved to '{output_path}'")
    
    # Step 2: Build BM25 index
    print("\n" + "=" * 60)
    print("Step 2: Building BM25 Index")
    print("=" * 60)
    
    pipeline = RankingPipeline(model_dir='models')
    pipeline.build_bm25_index(jobs_df)
    print("✓ BM25 index saved")
    
    # Step 3: Build semantic embeddings
    print("\n" + "=" * 60)
    print("Step 3: Building Semantic Embeddings")
    print("=" * 60)
    print("⚠ This may take several minutes (but faster on small sample)...")
    
    pipeline.build_semantic_embeddings(jobs_df)
    print("✓ Semantic embeddings saved")
    
    # Step 4: Train LTR model
    print("\n" + "=" * 60)
    print("Step 4: Training Learning-to-Rank Model")
    print("=" * 60)
    
    trainer = LTRTrainer(pipeline)
    X, y, qids = trainer.generate_synthetic_training_data(
        jobs_df,
        num_queries=min(100, len(jobs_df) // 5)  # fewer queries for small samples
    )
    print(f"Generated {len(X):,} training examples from {len(set(qids))} queries")
    
    model = trainer.train_model(X, y, qids)
    
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    metrics = trainer.evaluate_model(model, X, y, qids)
    
    print("\n✓ LTR model trained and saved")
    
    # Optional: skip ranking comparison in sample mode
    if not args.sample:
        compare = input("\nRun ranking comparison? (y/n): ").lower().strip()
        if compare == 'y':
            trainer.compare_ranking_stages(jobs_df, num_test_queries=50)
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  ✓ {output_path}")
    print("  ✓ models/bm25_index.pkl")
    print("  ✓ models/job_embeddings.npy")
    print("  ✓ models/ltr_model.pkl")
    print("  ✓ models/feature_names.pkl")
    
    print("\n🚀 You can now run the application:")
    print("   streamlit run app.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
