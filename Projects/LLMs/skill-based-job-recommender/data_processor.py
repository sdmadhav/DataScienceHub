import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import re
from tqdm import tqdm
from skill_extractor import SkillExtractor

class DataProcessor:
    def __init__(self, data_dir: str = 'data'):
        """Initialize data processor."""
        self.data_dir = Path(data_dir)
        self.skill_extractor = SkillExtractor()
        self.jobs_df = None
        self.companies_df = None
        self.benefits_df = None
        self.employee_counts_df = None
        
    def load_data(self):
        """Load all CSV files."""
        print("Loading data...")
        
        # Load main job postings
        self.jobs_df = pd.read_csv(self.data_dir / 'job_postings.csv')
        print(f"Loaded {len(self.jobs_df)} job postings")
        
        # Load companies
        companies_path = self.data_dir / 'company_details' / 'companies.csv'
        if companies_path.exists():
            self.companies_df = pd.read_csv(companies_path)
            print(f"Loaded {len(self.companies_df)} companies")
        
        # Load benefits
        benefits_path = self.data_dir / 'job_details' / 'benefits.csv'
        if benefits_path.exists():
            self.benefits_df = pd.read_csv(benefits_path)
            print(f"Loaded {len(self.benefits_df)} benefit records")
        
        # Load employee counts
        employee_path = self.data_dir / 'company_details' / 'employee_counts.csv'
        if employee_path.exists():
            self.employee_counts_df = pd.read_csv(employee_path)
            print(f"Loaded {len(self.employee_counts_df)} employee count records")
    
    def process_data(self) -> pd.DataFrame:
        """Process and merge all data."""
        print("Processing data...")

        # Merge with companies
        if self.companies_df is not None:
            df = self.jobs_df.merge(
                self.companies_df.rename(
                    columns={'name': 'company_name', 'description': 'company_description'}
                )[['company_id', 'company_name', 'company_description', 'company_size', 'country', 'state', 'city']],
                on='company_id',
                how='left'
            )
        else:
            df = self.jobs_df.copy()

        # Aggregate benefits per job
        if self.benefits_df is not None:
            benefits_agg = self.benefits_df.groupby('job_id')['type'].apply(list).reset_index()
            benefits_agg.columns = ['job_id', 'benefits_list']
            df = df.merge(benefits_agg, on='job_id', how='left')
        else:
            df['benefits_list'] = None

        # Merge employee counts
        if self.employee_counts_df is not None:
            latest_counts = self.employee_counts_df.sort_values('time_recorded').groupby('company_id').tail(1)
            df = df.merge(
                latest_counts[['company_id', 'employee_count', 'follower_count']],
                on='company_id',
                how='left'
            )

        # ✅ Drop duplicate columns before continuing
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Continue processing...
        print("Extracting skills from job descriptions...")
        # Extract skills
        print("Extracting skills from job descriptions...")
        # df['extracted_skills'] = df.apply(
        #     lambda row: self._extract_job_skills(row), axis=1
        # )
        # Extract skills in batch (MUCH FASTER)
        df['extracted_skills'] = self._extract_skills_batch(df)
        
        # Normalize experience levels
        df['experience_level_normalized'] = df['formatted_experience_level'].apply(self._normalize_experience)
        
        # Normalize salary to yearly USD (approximate)
        df['salary_yearly_usd'] = df.apply(self._normalize_salary, axis=1)
        
        # Parse location
        df['city_parsed'] = df['location'].apply(self._extract_city)
        df['state_parsed'] = df['location'].apply(self._extract_state)
        df['is_remote'] = df['remote_allowed'].fillna(0).astype(int)
        
        # Create searchable text
        df['searchable_text'] = df.apply(self._create_searchable_text, axis=1)
        
        # Handle missing values
        df['views'] = df['views'].fillna(0).astype(int)
        df['applies'] = df['applies'].fillna(0).astype(int)
        df['company_size'] = df['company_size'].fillna(-1).astype(int)
        
        # Convert benefits list to string for storage
        df['benefits_str'] = df['benefits_list'].apply(
            lambda x: ','.join(x) if isinstance(x, list) else ''
        )
        
        print(f"Processed {len(df)} jobs")
        return df

    def _extract_skills_batch(self, df: pd.DataFrame) -> pd.Series:
        """Extract skills in batch for all jobs with progress tracking."""
        
        print("Preparing texts for skill extraction...")
        # Combine description and skills_desc
        texts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing texts"):
            parts = []
            if pd.notna(row.get('description')):
                parts.append(str(row['description'])[:5000])  # Limit length
            if pd.notna(row.get('skills_desc')):
                parts.append(str(row['skills_desc'])[:2000])
            texts.append(' '.join(parts) if parts else '')
        
        print(f"Extracting skills from {len(texts)} job descriptions...")
        all_skills = self.skill_extractor.extract_skills_batch(texts)
        
        return pd.Series([','.join(sorted(skills)) for skills in all_skills])
      
    def _extract_job_skills(self, row) -> str:
        """Extract skills from job description and skills_desc."""
        skills = set()
        
        # Extract from description
        if pd.notna(row.get('description')):
            skills.update(self.skill_extractor.extract_skills(row['description']))
        
        # Extract from skills_desc
        if pd.notna(row.get('skills_desc')):
            skills.update(self.skill_extractor.extract_skills(row['skills_desc']))
        
        return ','.join(sorted(skills))
    
    def _normalize_experience(self, exp_level) -> int:
        """Normalize experience level to numeric scale."""
        if pd.isna(exp_level):
            return 2  # Default to mid-level
        
        exp_lower = str(exp_level).lower()
        if 'intern' in exp_lower or 'entry' in exp_lower:
            return 0
        elif 'associate' in exp_lower:
            return 1
        elif 'mid' in exp_lower:
            return 2
        elif 'senior' in exp_lower or 'lead' in exp_lower:
            return 3
        elif 'director' in exp_lower or 'executive' in exp_lower or 'principal' in exp_lower:
            return 4
        return 2
    
    def _normalize_salary(self, row) -> Optional[float]:
        """Normalize salary to yearly USD."""
        med_salary = row.get('med_salary')
        pay_period = row.get('pay_period')
        currency = row.get('currency')
        
        if pd.isna(med_salary) or med_salary == 0:
            return None
        
        # Convert to yearly
        salary_yearly = med_salary
        if pd.notna(pay_period):
            period_lower = str(pay_period).lower()
            if 'hour' in period_lower:
                salary_yearly = med_salary * 2080  # 40 hours/week * 52 weeks
            elif 'month' in period_lower:
                salary_yearly = med_salary * 12
        
        # Convert to USD (rough approximations)
        if pd.notna(currency):
            currency = str(currency).upper()
            conversion_rates = {
                'USD': 1.0,
                'EUR': 1.1,
                'GBP': 1.25,
                'CAD': 0.75,
                'AUD': 0.65,
                'INR': 0.012,
                'SGD': 0.74
            }
            salary_yearly *= conversion_rates.get(currency, 1.0)
        
        return salary_yearly
    
    def _extract_city(self, location) -> str:
        """Extract city from location string."""
        if pd.isna(location):
            return ''
        location_parts = str(location).split(',')
        return location_parts[0].strip() if location_parts else ''
    
    def _extract_state(self, location) -> str:
        """Extract state from location string."""
        if pd.isna(location):
            return ''
        location_parts = str(location).split(',')
        if len(location_parts) >= 2:
            state = location_parts[1].strip()
            # Extract state code (e.g., "CA" from "CA 12345")
            state_match = re.match(r'([A-Z]{2})', state)
            if state_match:
                return state_match.group(1)
            return state
        return ''
    
    def _create_searchable_text(self, row) -> str:
        """Create combined searchable text."""
        parts = []
        
        # Title (weighted heavily)
        if pd.notna(row.get('title')):
            parts.append(str(row['title']) * 3)  # Repeat for weight
        
        # Description
        if pd.notna(row.get('description')):
            parts.append(str(row['description']))
        
        # Skills
        if pd.notna(row.get('extracted_skills')):
            skills = str(row['extracted_skills']).replace(',', ' ')
            parts.append(skills * 2)  # Repeat for weight
        
        # Company name
        if pd.notna(row.get('company_name')):
            parts.append(str(row['company_name']))
        
        # Location
        if pd.notna(row.get('location')):
            parts.append(str(row['location']))
        
        return ' '.join(parts)
    
    def get_skill_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about skills in the dataset."""
        all_skills = []
        for skills_str in df['extracted_skills'].dropna():
            if skills_str:
                all_skills.extend(skills_str.split(','))
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        
        return {
            'total_jobs': len(df),
            'jobs_with_skills': df['extracted_skills'].notna().sum(),
            'unique_skills': len(skill_counts),
            'top_skills': skill_counts.most_common(20)
        }
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str = 'data/processed_jobs.parquet'):
        """Save processed data."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, input_path: str = 'data/processed_jobs.parquet') -> pd.DataFrame:
        """Load processed data."""
        return pd.read_parquet(input_path)