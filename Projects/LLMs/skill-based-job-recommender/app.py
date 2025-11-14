import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle

from skill_extractor import SkillExtractor
from ranking_pipeline import RankingPipeline

# Page config
st.set_page_config(
    page_title="Skills-Based Job Matching System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .skill-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .skill-match {
        background-color: #dcfce7;
        color: #166534;
    }
    .skill-missing {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        color: #1e40af;
    }
    .job-card {
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'skills': [],
        'experience_level': 2,
        'min_salary': 0,
        'city': '',
        'state': '',
        'remote_only': False,
        'company_size': -1,
        'benefits': []
    }

if 'search_results' not in st.session_state:
    st.session_state.search_results = None

if 'selected_job' not in st.session_state:
    st.session_state.selected_job = None

# Cache data loading
@st.cache_resource
def load_models_and_data():
    """Load all models and data."""
    # Load processed data
    jobs_df = pd.read_parquet('data/processed_jobs.parquet')
    
    # Initialize pipeline
    pipeline = RankingPipeline(model_dir='models')
    pipeline.jobs_df = jobs_df
    pipeline.load_indices()
    pipeline.load_ltr_model()
    
    # Load skill extractor
    skill_extractor = SkillExtractor()
    
    return pipeline, jobs_df, skill_extractor

try:
    pipeline, jobs_df, skill_extractor = load_models_and_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run preprocessing first: python preprocess.py")
    data_loaded = False

# Sidebar - User Profile
with st.sidebar:
    st.markdown("### 👤 User Profile")
    
    # Skills input
    st.markdown("#### Skills")
    all_skills = skill_extractor.get_all_skills() if data_loaded else []
    
    selected_skills = st.multiselect(
        "Select your skills",
        options=all_skills,
        default=st.session_state.user_profile['skills'],
        help="Start typing to search skills"
    )
    st.session_state.user_profile['skills'] = selected_skills
    
    # Display selected skills
    if selected_skills:
        st.markdown("**Your Skills:**")
        for skill in selected_skills:
            st.markdown(f'<span class="skill-badge skill-match">{skill}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Experience level
    st.markdown("#### Experience Level")
    exp_options = {
        0: "Entry Level",
        1: "Associate",
        2: "Mid-Level",
        3: "Senior",
        4: "Director/Executive"
    }
    experience = st.select_slider(
        "Select your experience",
        options=list(exp_options.keys()),
        value=st.session_state.user_profile['experience_level'],
        format_func=lambda x: exp_options[x]
    )
    st.session_state.user_profile['experience_level'] = experience
    
    st.markdown("---")
    
    # Location
    st.markdown("#### Location Preferences")
    city = st.text_input("City", value=st.session_state.user_profile['city'])
    state = st.text_input("State (2-letter code)", value=st.session_state.user_profile['state'], max_chars=2)
    remote_only = st.checkbox("Remote only", value=st.session_state.user_profile['remote_only'])
    
    st.session_state.user_profile['city'] = city
    st.session_state.user_profile['state'] = state.upper() if state else ''
    st.session_state.user_profile['remote_only'] = remote_only
    
    st.markdown("---")
    
    # Salary
    st.markdown("#### Salary Requirements")
    min_salary = st.number_input(
        "Minimum Salary (USD/year)",
        min_value=0,
        max_value=500000,
        value=st.session_state.user_profile['min_salary'],
        step=10000
    )
    st.session_state.user_profile['min_salary'] = min_salary
    
    st.markdown("---")
    
    # Company size
    st.markdown("#### Company Size Preference")
    company_size_options = {
        -1: "No Preference",
        0: "1-10 employees",
        1: "11-50 employees",
        2: "51-200 employees",
        3: "201-500 employees",
        4: "501-1000 employees",
        5: "1001-5000 employees",
        6: "5001-10000 employees",
        7: "10000+ employees"
    }
    company_size = st.selectbox(
        "Preferred company size",
        options=list(company_size_options.keys()),
        index=list(company_size_options.keys()).index(st.session_state.user_profile['company_size']),
        format_func=lambda x: company_size_options[x]
    )
    st.session_state.user_profile['company_size'] = company_size

# Main content
st.markdown('<h1 class="main-header">💼 Skills-Based Job Matching System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find your perfect job match using AI-powered ranking</p>', unsafe_allow_html=True)

if not data_loaded:
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Job Search", "📊 Market Insights", "📈 Skill Gap Analysis", "ℹ️ About"])

# Tab 1: Job Search
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Search for Jobs")
        if st.session_state.user_profile['skills']:
            st.success(f"✓ Profile complete with {len(st.session_state.user_profile['skills'])} skills")
        else:
            st.warning("⚠ Add skills to your profile to get started")
    
    with col2:
        search_button = st.button("🔍 Find Matching Jobs", type="primary", use_container_width=True)
    
    if search_button and st.session_state.user_profile['skills']:
        with st.spinner("Searching and ranking jobs..."):
            results_df = pipeline.rank_jobs(st.session_state.user_profile, top_k=50)
            st.session_state.search_results = results_df
    
    # Display results
    if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
        results_df = st.session_state.search_results
        
        st.markdown(f"### Found {len(results_df)} Matching Jobs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_match = st.slider("Minimum Match Score (%)", 0, 100, 0)
        with col2:
            work_type_filter = st.multiselect(
                "Work Type",
                options=results_df['formatted_work_type'].dropna().unique().tolist(),
                default=[]
            )
        with col3:
            exp_filter = st.multiselect(
                "Experience Level",
                options=results_df['formatted_experience_level'].dropna().unique().tolist(),
                default=[]
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        match_pct = filtered_df['skill_jaccard'] * 100
        filtered_df = filtered_df[match_pct >= min_match]
        
        if work_type_filter:
            filtered_df = filtered_df[filtered_df['formatted_work_type'].isin(work_type_filter)]
        if exp_filter:
            filtered_df = filtered_df[filtered_df['formatted_experience_level'].isin(exp_filter)]
        
        st.markdown(f"**Showing {len(filtered_df)} jobs**")
        
        # Display jobs
        for idx, row in filtered_df.head(20).iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"#### {row['title']}")
                    st.markdown(f"**{row.get('company_name', 'Unknown Company')}** • {row.get('location', 'Location not specified')}")
                    
                    # Match score
                    match_score = row['skill_jaccard'] * 100
                    st.markdown(f'<span class="match-score">{match_score:.0f}%</span> Match', unsafe_allow_html=True)
                    
                    # Skills
                    analysis = pipeline.analyze_match(st.session_state.user_profile, row)
                    
                    if analysis['matching_skills']:
                        st.markdown("**Matching Skills:**")
                        skills_html = ''.join([f'<span class="skill-badge skill-match">{s}</span>' for s in analysis['matching_skills'][:10]])
                        st.markdown(skills_html, unsafe_allow_html=True)
                    
                    if analysis['missing_skills']:
                        with st.expander(f"Missing Skills ({len(analysis['missing_skills'])})"):
                            skills_html = ''.join([f'<span class="skill-badge skill-missing">{s}</span>' for s in analysis['missing_skills'][:10]])
                            st.markdown(skills_html, unsafe_allow_html=True)
                
                with col2:
                    # Salary
                    if pd.notna(row.get('salary_yearly_usd')) and row['salary_yearly_usd'] > 0:
                        st.metric("Salary", f"${row['salary_yearly_usd']:,.0f}")
                    
                    # Work type
                    if pd.notna(row.get('formatted_work_type')):
                        st.markdown(f"**Type:** {row['formatted_work_type']}")
                    
                    # View details button
                    if st.button("View Details", key=f"view_{row['job_id']}"):
                        st.session_state.selected_job = row
                        st.rerun()
                
                st.markdown("---")

# Tab 2: Market Insights
with tab2:
    st.markdown("### 📊 Job Market Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top skills
        st.markdown("#### Most In-Demand Skills")
        all_skills_list = []
        for skills_str in jobs_df['extracted_skills'].dropna():
            if skills_str:
                all_skills_list.extend(skills_str.split(','))
        
        from collections import Counter
        skill_counts = Counter(all_skills_list)
        top_skills = skill_counts.most_common(15)
        
        if top_skills:
            skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
            fig = px.bar(skills_df, x='Count', y='Skill', orientation='h',
                        title="Top 15 Most Demanded Skills",
                        color='Count', color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary distribution
        st.markdown("#### Salary Distribution")
        salary_data = jobs_df[jobs_df['salary_yearly_usd'] > 0]['salary_yearly_usd']
        if len(salary_data) > 0:
            fig = px.histogram(salary_data, nbins=50, title="Salary Distribution (USD/year)")
            fig.update_layout(xaxis_title="Salary", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Median Salary", f"${salary_data.median():,.0f}")
            st.metric("Average Salary", f"${salary_data.mean():,.0f}")
    
    # Experience level distribution
    st.markdown("#### Jobs by Experience Level")
    exp_counts = jobs_df['formatted_experience_level'].value_counts()
    if len(exp_counts) > 0:
        fig = px.pie(values=exp_counts.values, names=exp_counts.index, 
                    title="Job Distribution by Experience Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # Location insights
    st.markdown("#### Top Locations")
    location_counts = jobs_df['location'].value_counts().head(10)
    if len(location_counts) > 0:
        fig = px.bar(x=location_counts.values, y=location_counts.index, orientation='h',
                    title="Top 10 Job Locations")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Skill Gap Analysis
with tab3:
    st.markdown("### 📈 Skill Gap Analysis")
    
    if st.session_state.user_profile['skills']:
        st.markdown("#### Based on Your Profile")
        
        # Analyze gap for top jobs
        if st.session_state.search_results is not None and len(st.session_state.search_results) > 0:
            top_jobs = st.session_state.search_results.head(10)
            
            all_missing = []
            for _, job in top_jobs.iterrows():
                analysis = pipeline.analyze_match(st.session_state.user_profile, job)
                all_missing.extend(analysis['missing_skills'])
            
            from collections import Counter
            missing_counts = Counter(all_missing)
            
            if missing_counts:
                st.markdown("#### Skills to Learn for Better Matches")
                st.markdown("These skills appear most frequently in jobs you're not fully qualified for:")
                
                top_missing = missing_counts.most_common(10)
                missing_df = pd.DataFrame(top_missing, columns=['Skill', 'Frequency'])
                
                fig = px.bar(missing_df, x='Frequency', y='Skill', orientation='h',
                            color='Frequency', color_continuous_scale='Reds')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("#### 🎯 Recommendations")
                for skill, freq in top_missing[:5]:
                    st.success(f"**{skill}** - Found in {freq} relevant jobs")
            else:
                st.success("Great! You match well with top jobs.")
        else:
            st.info("Search for jobs first to see your skill gap analysis")
    else:
        st.warning("Add skills to your profile to see skill gap analysis")

# Tab 4: About
with tab4:
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    This Skills-Based Job Matching System uses a sophisticated multi-stage ranking pipeline:
    
    #### 🔍 Stage 1: BM25 Retrieval
    - Fast candidate retrieval using BM25 algorithm
    - Retrieves top 100 candidates from 124K+ job postings
    
    #### 🧠 Stage 2: Semantic Search
    - Uses sentence transformers for semantic similarity
    - Reranks candidates using deep learning embeddings
    - Narrows down to top 50 matches
    
    #### 🎯 Stage 3: Learning-to-Rank
    - LightGBM LambdaRank model
    - Considers 20+ features including:
        - Skill match (Jaccard similarity)
        - Experience level alignment
        - Location match
        - Salary fit
        - Company size preference
        - Benefits match
        - Job popularity signals
    
    #### 📊 Features
    - 100+ skills across multiple categories
    - Query expansion with related skills
    - Real-time ranking and matching
    - Detailed skill gap analysis
    - Market insights and trends
    
    #### 🔧 Technology Stack
    - **Backend**: Python, Streamlit
    - **IR**: BM25, Sentence Transformers
    - **ML**: LightGBM, scikit-learn
    - **NLP**: spaCy
    - **Visualization**: Plotly
    
    #### 📝 Dataset
    LinkedIn Job Postings (2023-2024) from Kaggle - 124,000+ job listings
    """)

# Job Detail Modal (using expander)
if st.session_state.selected_job is not None:
    job = st.session_state.selected_job
    
    with st.expander("📋 Job Details", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"## {job['title']}")
            st.markdown(f"### {job.get('company_name', 'Unknown Company')}")
            
            st.markdown("#### Description")
            description = job.get('description', 'No description available')
            st.markdown(description[:1000] + "..." if len(str(description)) > 1000 else description)
            
            # Skills breakdown
            st.markdown("#### Required Skills")
            analysis = pipeline.analyze_match(st.session_state.user_profile, job)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Skills Matched", f"{analysis['total_matched']}/{analysis['total_required']}")
            with col_b:
                st.metric("Match Percentage", f"{analysis['match_percentage']:.0f}%")
            
            if analysis['matching_skills']:
                st.success("**You Have:**")
                skills_html = ''.join([f'<span class="skill-badge skill-match">{s}</span>' for s in analysis['matching_skills']])
                st.markdown(skills_html, unsafe_allow_html=True)
            
            if analysis['missing_skills']:
                st.error("**You Need:**")
                skills_html = ''.join([f'<span class="skill-badge skill-missing">{s}</span>' for s in analysis['missing_skills']])
                st.markdown(skills_html, unsafe_allow_html=True)
        
        with col2:
            # Job details
            st.markdown("#### Job Information")
            
            if pd.notna(job.get('salary_yearly_usd')) and job['salary_yearly_usd'] > 0:
                st.metric("Salary", f"${job['salary_yearly_usd']:,.0f}/year")
            
            if pd.notna(job.get('formatted_work_type')):
                st.markdown(f"**Work Type:** {job['formatted_work_type']}")
            
            if pd.notna(job.get('formatted_experience_level')):
                st.markdown(f"**Experience:** {job['formatted_experience_level']}")
            
            st.markdown(f"**Location:** {job.get('location', 'Not specified')}")
            
            if job.get('is_remote', 0) == 1:
                st.success("✓ Remote Available")
            
            # Company info
            st.markdown("#### Company Info")
            if pd.notna(job.get('employee_count')):
                st.markdown(f"**Employees:** {int(job['employee_count']):,}")
            
            if pd.notna(job.get('follower_count')):
                st.markdown(f"**LinkedIn Followers:** {int(job['follower_count']):,}")
            
            # Application stats
            st.markdown("#### Application Stats")
            st.markdown(f"**Views:** {int(job.get('views', 0)):,}")
            st.markdown(f"**Applications:** {int(job.get('applies', 0)):,}")
            
            # Apply button
            if pd.notna(job.get('job_posting_url')):
                st.link_button("Apply on LinkedIn", job['job_posting_url'], use_container_width=True)
        
        if st.button("Close Details"):
            st.session_state.selected_job = None
            st.rerun()