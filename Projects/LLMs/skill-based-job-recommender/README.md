# Skills-Based Job Matching System

A sophisticated job matching system using multi-stage Information Retrieval and Learning-to-Rank techniques. Matches job seekers with relevant positions based on skills, experience, location, salary, and more.

![System Architecture](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

### Multi-Stage Ranking Pipeline
1. **BM25 Retrieval** - Fast candidate retrieval from 124K+ jobs
2. **Semantic Search** - Deep learning-based reranking with sentence transformers
3. **Learning-to-Rank** - LightGBM LambdaRank model with 20+ features

### User Features
- **Skill-based Matching** - 100+ skills across multiple categories
- **Query Expansion** - Automatically expands searches with related skills
- **Skill Gap Analysis** - Identifies missing skills for target roles
- **Market Insights** - Visualize trending skills and salary distributions
- **Detailed Job Analysis** - View comprehensive match breakdowns

### Technical Highlights
- Processes 124,000+ LinkedIn job postings (2023-2024)
- Real-time search with <2 second response time
- Comprehensive skill taxonomy with NLP-based extraction
- Interactive visualizations with Plotly
- Clean, modern UI with Streamlit

## 📋 Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended for preprocessing)
- 2GB disk space for data and models

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd job-matching-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download Dataset

Download the LinkedIn Job Postings dataset from Kaggle:

**Dataset**: [LinkedIn Job Postings (2023-2024)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

```bash
# Using Kaggle API
kaggle datasets download -d arshkon/linkedin-job-postings

# Extract to data/ directory
unzip linkedin-job-postings.zip -d data/
```

Your directory structure should look like:
```
data/
├── job_postings.csv
├── job_details/
│   └── benefits.csv
└── company_details/
    ├── companies.csv
    └── employee_counts.csv
```

### 4. Preprocess Data

This step loads the data, extracts skills, builds indices, and trains the LTR model:

```bash
python preprocess.py
```

**Expected time**: 15-30 minutes depending on hardware

This creates:
- `data/processed_jobs.parquet` - Processed job data with extracted skills
- `models/bm25_index.pkl` - BM25 search index
- `models/job_embeddings.npy` - Semantic embeddings (1.5GB)
- `models/ltr_model.pkl` - Trained LightGBM model
- `models/feature_names.pkl` - Feature metadata

### 5. Run Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
job-matching-system/
├── app.py                   # Streamlit application
├── skill_extractor.py       # Skill taxonomy and extraction
├── data_processor.py        # Data loading and preprocessing
├── ranking_pipeline.py      # Multi-stage ranking system
├── ltr_trainer.py          # Learning-to-Rank training
├── preprocess.py           # Data preprocessing script
├── evaluation.py           # Evaluation and metrics
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Data directory (not in repo, get from KAGGLE link below)
│   ├── job_postings.csv
│   ├── processed_jobs.parquet
│   ├── job_details/
│   └── company_details/
│
├── models/               # Generated models (not in repo)
│   ├── bm25_index.pkl
│   ├── job_embeddings.npy
│   ├── ltr_model.pkl
│   └── feature_names.pkl
│
└── evaluation_results/   # Evaluation outputs
    ├── feature_importance.csv
    ├── stage_comparison.csv
    └── example_queries.csv
```

## 🔧 System Architecture

### Data Processing Pipeline

```
Raw CSV Files
    ↓
Data Processor
    ├── Merge jobs, companies, benefits
    ├── Extract skills using NLP
    ├── Normalize salary and experience
    ├── Parse location data
    └── Create searchable text
    ↓
Processed DataFrame
```

### Skill Extraction

```python
# Supports 100+ skills across categories:
- Programming Languages (Python, Java, JavaScript, etc.)
- Web Development (React, Angular, Node.js, etc.)
- Databases (SQL, MongoDB, PostgreSQL, etc.)
- Cloud & DevOps (AWS, Docker, Kubernetes, etc.)
- Data Science & ML (TensorFlow, PyTorch, Pandas, etc.)
- Mobile (iOS, Android, React Native, etc.)
```

**Extraction Methods**:
1. Regex pattern matching with comprehensive taxonomy
2. spaCy Named Entity Recognition
3. Multi-word skill detection

### Multi-Stage Ranking

#### Stage 1: BM25 Retrieval
- **Algorithm**: Okapi BM25
- **Input**: User skills + preferences
- **Output**: Top 100 candidates
- **Speed**: <100ms

#### Stage 2: Semantic Search
- **Model**: sentence-transformers (all-MiniLM-L6-v2)
- **Input**: Top 100 from BM25
- **Output**: Top 50 reranked by semantic similarity
- **Speed**: ~200ms

#### Stage 3: Learning-to-Rank
- **Model**: LightGBM LambdaRank
- **Features** (20+):
  - `skill_jaccard` - Jaccard similarity of skills
  - `skill_overlap_pct` - Percentage of job skills matched
  - `num_matching_skills` - Count of matching skills
  - `experience_match` - Experience level alignment (0-1)
  - `location_score` - City/state/remote match
  - `salary_match` - Salary fit score
  - `company_size_match` - Company size preference
  - `benefits_match` - Benefits overlap
  - `log_applies`, `log_views` - Popularity signals
  - `bm25_score`, `semantic_score` - Previous stage scores
- **Output**: Final ranked top 50 results
- **Speed**: ~50ms

### Query Expansion

Automatically expands user skills with related technologies:

```python
User searches: ['python', 'aws']
Expanded to: ['python', 'django', 'flask', 'pandas', 'aws', 'docker', 'kubernetes']
```

## 💻 Usage Guide

### 1. Set Up Profile

In the sidebar, configure your profile:

- **Skills**: Select from 100+ available skills (autocomplete enabled)
- **Experience Level**: Entry to Director/Executive
- **Location**: City, state, and remote preference
- **Salary**: Minimum acceptable salary (USD/year)
- **Company Size**: Optional preference
- **Benefits**: Preferred benefits (optional)

### 2. Search for Jobs

Click "Find Matching Jobs" to run the full ranking pipeline. Results show:

- **Match Score**: Overall percentage match (0-100%)
- **Job Details**: Title, company, location, salary
- **Skill Breakdown**: Matching vs. missing skills
- **Quick Actions**: View details, apply

### 3. Filter Results

Refine results using:
- Minimum match score threshold
- Work type (Full-time, Part-time, Contract)
- Experience level

### 4. Analyze Skill Gaps

The **Skill Gap Analysis** tab shows:
- Most frequently missing skills in target roles
- Recommendations for skills to learn
- Frequency analysis across top matches

### 5. Explore Market Insights

View market trends:
- Most in-demand skills (bar chart)
- Salary distributions by role
- Job counts by experience level
- Top hiring locations

### 6. View Job Details

Click "View Details" on any job to see:
- Full job description
- Complete skill breakdown with match visualization
- Company information and stats
- Application statistics
- Direct apply link

## 📊 Evaluation

Run comprehensive evaluation:

```bash
python evaluation.py
```

This generates:

### 1. NDCG Metrics
- NDCG@1, @3, @5, @10, @20
- Measures ranking quality

### 2. Feature Importance
- Top contributing features
- Saved to `evaluation_results/feature_importance.csv`

### 3. Stage Comparison
- Performance of BM25 only vs. BM25+Semantic vs. Full pipeline
- Saved to `evaluation_results/stage_comparison.csv`

### 4. Example Queries
- Predefined test queries with results
- Saved to `evaluation_results/example_queries.csv`

## 🎨 Customization

### Adding New Skills

Edit `skill_extractor.py`:

```python
self.skill_taxonomy = {
    'new_skill': ['new_skill', 'alias1', 'alias2'],
    # Add more skills...
}
```

### Adjusting Ranking Weights

In `ranking_pipeline.py`, modify the fallback scoring:

```python
ltr_score = (
    features['skill_jaccard'] * 0.3 +      # Skill match weight
    features['semantic_score'] * 0.25 +    # Semantic weight
    features['experience_match'] * 0.15 +  # Experience weight
    # Adjust weights as needed
)
```

### Changing UI Theme

Modify the CSS in `app.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #your-color;  /* Change colors */
    }
</style>
""", unsafe_allow_html=True)
```

## 🔍 API Reference

### RankingPipeline

```python
pipeline = RankingPipeline(model_dir='models')
pipeline.load_indices()  # Load BM25 and embeddings
pipeline.load_ltr_model()  # Load trained model

# Search for jobs
results = pipeline.rank_jobs(user_profile, top_k=50)

# Analyze specific match
analysis = pipeline.analyze_match(user_profile, job_row)
```

### SkillExtractor

```python
extractor = SkillExtractor()

# Extract skills from text
skills = extractor.extract_skills(job_description)

# Expand query
expanded = extractor.expand_query_skills(user_skills)

# Get all available skills
all_skills = extractor.get_all_skills()
```

## 📈 Performance Metrics

Based on synthetic evaluation data:

- **Average NDCG@10**: ~0.75-0.85
- **Search Speed**: <2 seconds for full pipeline
- **BM25 Retrieval**: <100ms
- **Semantic Reranking**: ~200ms
- **LTR Scoring**: ~50ms

## 🐛 Troubleshooting

### Issue: "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Memory error during preprocessing"
- Reduce batch size in `ranking_pipeline.py`:
  ```python
  batch_size = 16  # Reduce from 32
  ```
- Process data in chunks

### Issue: "Slow search performance"
- Ensure models are properly cached
- Check that indices are loaded (not rebuilt each time)
- Use `@st.cache_resource` for expensive operations

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt --upgrade
```

## 🚦 Limitations

- **Dataset**: Limited to LinkedIn jobs from 2023-2024
- **Skill Extraction**: May miss domain-specific or emerging skills
- **Salary Normalization**: Approximate currency conversions
- **Company Data**: Not all jobs have complete company information
- **Real-time Data**: Dataset is static, not live job postings

## 🔮 Future Enhancements

- [ ] Real-time job scraping integration
- [ ] User feedback learning
- [ ] Resume parsing and auto-profile creation
- [ ] Career path recommendations
- [ ] Interview preparation resources
- [ ] Salary negotiation insights
- [ ] Job alert notifications
- [ ] Multi-language support

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) by Arsh Koneru
- Sentence Transformers by UKPLab
- LightGBM by Microsoft
- Streamlit framework

## 📧 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Built with ❤️ using Python, Streamlit, and modern IR techniques**
