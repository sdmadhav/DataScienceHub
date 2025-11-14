import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import json

# Import your existing modules
from skill_extractor import SkillExtractor
from ranking_pipeline import RankingPipeline

# Load data and models
def load_system():
    """Load all models and data."""
    try:
        print("Loading jobs data...")
        jobs_df = pd.read_parquet('data/processed_jobs.parquet')
        print(f"✓ Loaded {len(jobs_df)} jobs")
        
        print("Initializing pipeline...")
        pipeline = RankingPipeline(model_dir='models')
        pipeline.jobs_df = jobs_df
        
        print("Loading indices...")
        pipeline.load_indices()
        print("✓ Indices loaded")
        
        print("Loading LTR model...")
        pipeline.load_ltr_model()
        print("✓ LTR model loaded")
        
        print("Loading skill extractor...")
        skill_extractor = SkillExtractor()
        print("✓ Skill extractor loaded")
        
        return pipeline, jobs_df, skill_extractor, True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False

print("="*50)
print("INITIALIZING SYSTEM...")
print("="*50)
pipeline, jobs_df, skill_extractor, data_loaded = load_system()
all_skills = skill_extractor.get_all_skills() if data_loaded else []
print("="*50)
print(f"✓ SYSTEM READY - {len(all_skills)} skills loaded")
print("="*50)

# Global state for user profile
user_profile = {
    'skills': [],
    'experience_level': 2,
    'min_salary': 0,
    'city': '',
    'state': '',
    'remote_only': False,
    'company_size': -1
}

def search_jobs(skills, experience, min_salary, city, state, remote_only, company_size):
    """Search and rank jobs based on user profile."""
    if not data_loaded:
        return "❌ System not loaded. Please run preprocessing first.", None, None
    
    if not skills:
        return "⚠️ Please select at least one skill to search for jobs.", None, None
    
    try:
        # Update user profile
        user_profile['skills'] = skills
        user_profile['experience_level'] = experience
        user_profile['min_salary'] = min_salary
        user_profile['city'] = city
        user_profile['state'] = state.upper() if state else ''
        user_profile['remote_only'] = remote_only
        user_profile['company_size'] = company_size
        
        # Search jobs
        results_df = pipeline.rank_jobs(user_profile, top_k=50)
        
        if len(results_df) == 0:
            return "No matching jobs found.", None, None
        
        # Create results HTML
        html_results = create_results_html(results_df, user_profile)
        
        # Create insights plot
        skills_plot = create_skills_plot(results_df)
        salary_plot = create_salary_plot(results_df)
        
        return html_results, skills_plot, salary_plot
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during search: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None, None

def create_results_html(results_df, profile):
    """Create HTML for job results."""
    html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h2 style="color: #1e40af;">🎯 Found {len(results_df)} Matching Jobs</h2>
    """
    
    for idx, row in results_df.head(15).iterrows():
        match_score = row['skill_jaccard'] * 100
        analysis = pipeline.analyze_match(profile, row)
        
        # Job card
        html += f"""
        <div style="border: 2px solid #e2e8f0; border-radius: 10px; padding: 20px; margin: 20px 0; background: white;">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div style="flex: 1;">
                    <h3 style="color: #1e40af; margin: 0 0 5px 0;">{row['title']}</h3>
                    <p style="color: #64748b; margin: 5px 0;">
                        <strong>{row.get('company_name_y', 'Unknown Company')}</strong> • 
                        {row.get('location', 'Location not specified')}
                    </p>
                    
                    <div style="margin: 15px 0;">
                        <span style="font-size: 32px; font-weight: bold; color: #1e40af;">{match_score:.0f}%</span>
                        <span style="color: #64748b;"> Match</span>
                    </div>
                    
                    <div style="margin: 10px 0;">
                        <strong style="color: #16a34a;">✓ Matching Skills ({len(analysis['matching_skills'])}):</strong><br>
        """
        
        # Matching skills badges
        for skill in analysis['matching_skills'][:8]:
            html += f'<span style="display: inline-block; background: #dcfce7; color: #166534; padding: 5px 12px; margin: 3px; border-radius: 15px; font-size: 13px;">{skill}</span>'
        
        if len(analysis['matching_skills']) > 8:
            html += f'<span style="color: #64748b; font-size: 13px;"> +{len(analysis["matching_skills"]) - 8} more</span>'
        
        html += "</div>"
        
        # Missing skills
        if analysis['missing_skills']:
            html += f"""
                    <div style="margin: 10px 0;">
                        <strong style="color: #dc2626;">✗ Missing Skills ({len(analysis['missing_skills'])}):</strong><br>
            """
            for skill in analysis['missing_skills'][:6]:
                html += f'<span style="display: inline-block; background: #fee2e2; color: #991b1b; padding: 5px 12px; margin: 3px; border-radius: 15px; font-size: 13px;">{skill}</span>'
            
            if len(analysis['missing_skills']) > 6:
                html += f'<span style="color: #64748b; font-size: 13px;"> +{len(analysis["missing_skills"]) - 6} more</span>'
            html += "</div>"
        
        html += """
                </div>
                <div style="text-align: right; padding-left: 20px;">
        """
        
        # Salary
        if pd.notna(row.get('salary_yearly_usd')) and row['salary_yearly_usd'] > 0:
            html += f'<div style="background: #f0f9ff; padding: 10px; border-radius: 8px; margin-bottom: 10px;"><strong style="color: #0369a1;">${row["salary_yearly_usd"]:,.0f}</strong><br><span style="font-size: 12px; color: #64748b;">per year</span></div>'
        
        # Work type
        if pd.notna(row.get('formatted_work_type')):
            html += f'<div style="margin: 5px 0;"><strong>Type:</strong> {row["formatted_work_type"]}</div>'
        
        # Experience
        if pd.notna(row.get('formatted_experience_level')):
            html += f'<div style="margin: 5px 0;"><strong>Level:</strong> {row["formatted_experience_level"]}</div>'
        
        # Remote
        if row.get('is_remote', 0) == 1:
            html += '<div style="margin: 5px 0; color: #16a34a;">🏠 Remote Available</div>'
        
        html += """
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

def create_skills_plot(results_df):
    """Create top skills visualization."""
    all_skills_list = []
    for _, row in results_df.iterrows():
        if pd.notna(row.get('extracted_skills')):
            all_skills_list.extend(row['extracted_skills'].split(','))
    
    skill_counts = Counter(all_skills_list)
    top_skills = skill_counts.most_common(10)
    
    if not top_skills:
        return None
    
    skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
    fig = px.bar(skills_df, x='Count', y='Skill', orientation='h',
                title="Top Skills in Matching Jobs",
                color='Count', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    return fig

def create_salary_plot(results_df):
    """Create salary distribution plot."""
    salary_data = results_df[results_df['salary_yearly_usd'] > 0]['salary_yearly_usd']
    
    if len(salary_data) == 0:
        return None
    
    fig = px.histogram(salary_data, nbins=20, 
                      title=f"Salary Distribution (Median: ${salary_data.median():,.0f})")
    fig.update_layout(xaxis_title="Salary (USD/year)", yaxis_title="Count", height=400)
    return fig

def create_market_insights():
    """Create market insights plots."""
    if not data_loaded:
        return None, None, None
    
    try:
        # Top skills overall
        all_skills_list = []
        for skills_str in jobs_df['extracted_skills'].dropna():
            if skills_str:
                all_skills_list.extend(skills_str.split(','))
        
        skill_counts = Counter(all_skills_list)
        top_skills = skill_counts.most_common(15)
        
        skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        skills_fig = px.bar(skills_df, x='Count', y='Skill', orientation='h',
                    title="Top 15 Most In-Demand Skills (All Jobs)",
                    color='Count', color_continuous_scale='Viridis')
        skills_fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        
        # Salary distribution
        salary_data = jobs_df[jobs_df['salary_yearly_usd'] > 0]['salary_yearly_usd']
        salary_fig = px.histogram(salary_data, nbins=50, 
                                 title="Overall Salary Distribution")
        salary_fig.update_layout(xaxis_title="Salary (USD/year)", yaxis_title="Count", height=400)
        
        # Experience level distribution
        exp_counts = jobs_df['formatted_experience_level'].value_counts()
        exp_fig = px.pie(values=exp_counts.values, names=exp_counts.index,
                        title="Jobs by Experience Level", height=400)
        
        return skills_fig, salary_fig, exp_fig
    except Exception as e:
        print(f"Error creating market insights: {e}")
        return None, None, None

# Create Gradio interface - REMOVE theme parameter that's causing issues
with gr.Blocks(title="Skills-Based Job Matching System") as demo:
    gr.Markdown("""
    # 💼 Skills-Based Job Matching System
    ### Find your perfect job match using AI-powered ranking
    """)
    
    if not data_loaded:
        gr.Markdown("## ⚠️ SYSTEM NOT LOADED - Please check console for errors")
    
    with gr.Tabs():
        # Tab 1: Job Search
        with gr.Tab("🔍 Job Search"):
            gr.Markdown("### Search for Jobs Based on Your Profile")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 👤 Your Profile")
                    
                    skills_input = gr.Dropdown(
                        choices=all_skills,
                        multiselect=True,
                        label="Select Your Skills",
                        info="Choose skills you have",
                        interactive=True
                    )
                    
                    experience_input = gr.Slider(
                        minimum=0, maximum=4, step=1, value=2,
                        label="Experience Level",
                        info="0=Entry, 1=Associate, 2=Mid, 3=Senior, 4=Director"
                    )
                    
                    min_salary_input = gr.Number(
                        value=0, label="Minimum Salary (USD/year)",
                        info="Your minimum salary requirement"
                    )
                    
                    with gr.Row():
                        city_input = gr.Textbox(label="City", placeholder="e.g., San Francisco")
                        state_input = gr.Textbox(label="State", placeholder="CA", max_lines=1)
                    
                    remote_input = gr.Checkbox(label="Remote Only", value=False)
                    
                    company_size_input = gr.Slider(
                        minimum=-1, maximum=7, step=1, value=-1,
                        label="Company Size Preference (-1=No Preference, 0=1-10, 7=10000+)"
                    )
                    
                    search_btn = gr.Button("🔍 Find Matching Jobs", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    results_html = gr.HTML(label="Search Results")
            
            with gr.Row():
                skills_plot = gr.Plot(label="Top Skills in Results")
                salary_plot = gr.Plot(label="Salary Distribution")
            
            search_btn.click(
                fn=search_jobs,
                inputs=[skills_input, experience_input, min_salary_input, 
                       city_input, state_input, remote_input, company_size_input],
                outputs=[results_html, skills_plot, salary_plot]
            )
        
        # Tab 2: Market Insights
        with gr.Tab("📊 Market Insights"):
            gr.Markdown("### Overall Job Market Analysis")
            
            if data_loaded:
                insights_btn = gr.Button("Generate Market Insights", variant="primary")
                
                with gr.Row():
                    market_skills_plot = gr.Plot(label="Most In-Demand Skills")
                
                with gr.Row():
                    market_salary_plot = gr.Plot(label="Salary Distribution")
                    market_exp_plot = gr.Plot(label="Experience Level Distribution")
                
                insights_btn.click(
                    fn=create_market_insights,
                    outputs=[market_skills_plot, market_salary_plot, market_exp_plot]
                )
                
                # Stats
                with gr.Row():
                    gr.Markdown(f"""
                    ### 📈 Dataset Statistics
                    - **Total Jobs:** {len(jobs_df):,}
                    - **Unique Skills:** {len(all_skills)}
                    - **Companies:** {jobs_df['company_name_y'].nunique():,}
                    - **Median Salary:** ${jobs_df[jobs_df['salary_yearly_usd'] > 0]['salary_yearly_usd'].median():,.0f}
                    """)
            else:
                gr.Markdown("⚠️ Data not loaded. Please run preprocessing first.")
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This System
            
            This Skills-Based Job Matching System uses a sophisticated multi-stage ranking pipeline:
            
            ### 🔍 Stage 1: BM25 Retrieval
            - Fast candidate retrieval using BM25 algorithm
            - Retrieves top 100 candidates from 124K+ job postings
            
            ### 🧠 Stage 2: Semantic Search
            - Uses sentence transformers for semantic similarity
            - Reranks candidates using deep learning embeddings
            - Narrows down to top 50 matches
            
            ### 🎯 Stage 3: Learning-to-Rank
            - LightGBM LambdaRank model
            - Considers 20+ features including skill match, experience, location, salary, etc.
            
            ### 📊 Features
            - 100+ skills across multiple categories
            - Query expansion with related skills
            - Real-time ranking and matching
            - Detailed skill gap analysis
            - Market insights and trends
            """)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("LAUNCHING GRADIO INTERFACE...")
    print("="*50 + "\n")
    
    # Launch with optimized settings for Colab
    demo.launch(
        share=True,
        debug=False,  # Disable debug to reduce console noise
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )