import re
from tqdm import tqdm
import spacy
from typing import List, Set, Dict
from collections import defaultdict

class SkillExtractor:
    def __init__(self):
        """Initialize skill extractor with comprehensive taxonomy."""
        self.nlp = None
        self._load_spacy()
        
        # Comprehensive skill taxonomy organized by category
        self.skill_taxonomy = {
            # Programming Languages
            'python': ['python', 'python3', 'py'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6'],
            'java': ['java', 'java8', 'java11', 'java17'],
            'typescript': ['typescript', 'ts'],
            'cpp': ['c++', 'cpp', 'cplusplus'],
            'csharp': ['c#', 'csharp', '.net', 'dotnet'],
            'ruby': ['ruby', 'ruby on rails', 'ror', 'rails'],
            'go': ['go', 'golang'],
            'rust': ['rust'],
            'php': ['php', 'php7', 'php8'],
            'swift': ['swift', 'swiftui'],
            'kotlin': ['kotlin'],
            'scala': ['scala'],
            'r': ['r programming', 'r language'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
            
            # Web Frontend
            'react': ['react', 'reactjs', 'react.js', 'react native'],
            'angular': ['angular', 'angularjs', 'angular2'],
            'vue': ['vue', 'vuejs', 'vue.js', 'nuxt'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3', 'sass', 'scss', 'less'],
            'jquery': ['jquery'],
            'bootstrap': ['bootstrap'],
            'tailwind': ['tailwind', 'tailwindcss'],
            'webpack': ['webpack', 'vite', 'rollup'],
            
            # Backend & Frameworks
            'nodejs': ['node', 'nodejs', 'node.js', 'express', 'expressjs'],
            'django': ['django'],
            'flask': ['flask'],
            'fastapi': ['fastapi'],
            'spring': ['spring', 'spring boot', 'springboot'],
            'laravel': ['laravel'],
            'asp.net': ['asp.net', 'aspnet'],
            
            # Databases
            'mongodb': ['mongodb', 'mongo'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'mysql': ['mysql'],
            'redis': ['redis'],
            'elasticsearch': ['elasticsearch', 'elastic'],
            'cassandra': ['cassandra'],
            'dynamodb': ['dynamodb'],
            'oracle': ['oracle database', 'oracle db'],
            'mssql': ['sql server', 'mssql', 'microsoft sql'],
            
            # Cloud & DevOps
            'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'terraform': ['terraform'],
            'jenkins': ['jenkins'],
            'gitlab': ['gitlab', 'gitlab ci', 'gitlab-ci'],
            'github': ['github', 'github actions'],
            'circleci': ['circleci'],
            'ansible': ['ansible'],
            'chef': ['chef'],
            'puppet': ['puppet'],
            
            # Data Science & ML
            'machine learning': ['machine learning', 'ml', 'deep learning'],
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'scikit-learn': ['scikit-learn', 'sklearn'],
            'pandas': ['pandas'],
            'numpy': ['numpy'],
            'jupyter': ['jupyter', 'jupyter notebook'],
            'data science': ['data science', 'data analysis'],
            'nlp': ['nlp', 'natural language processing'],
            'computer vision': ['computer vision', 'cv', 'opencv'],
            'keras': ['keras'],
            'spark': ['apache spark', 'pyspark', 'spark'],
            'hadoop': ['hadoop', 'mapreduce'],
            
            # Mobile
            'ios': ['ios', 'ios development', 'iphone'],
            'android': ['android', 'android development'],
            'react native': ['react native'],
            'flutter': ['flutter', 'dart'],
            
            # Testing & QA
            'testing': ['testing', 'qa', 'quality assurance'],
            'selenium': ['selenium'],
            'jest': ['jest'],
            'pytest': ['pytest'],
            'junit': ['junit'],
            'cypress': ['cypress'],
            
            # Version Control
            'git': ['git', 'version control'],
            'svn': ['svn', 'subversion'],
            
            # Project Management & Methodologies
            'agile': ['agile', 'scrum', 'kanban'],
            'jira': ['jira'],
            'confluence': ['confluence'],
            'project management': ['project management', 'pmp'],
            
            # Soft Skills
            'leadership': ['leadership', 'team lead', 'management'],
            'communication': ['communication', 'presentation'],
            'problem solving': ['problem solving', 'analytical'],
            'collaboration': ['collaboration', 'teamwork'],
            
            # Business & Analytics
            'tableau': ['tableau'],
            'power bi': ['power bi', 'powerbi'],
            'excel': ['excel', 'microsoft excel'],
            'salesforce': ['salesforce', 'crm'],
            'sap': ['sap'],
            
            # Security
            'security': ['security', 'cybersecurity', 'infosec'],
            'penetration testing': ['penetration testing', 'pentesting'],
            'encryption': ['encryption', 'cryptography'],
            
            # Other Technical
            'rest api': ['rest', 'rest api', 'restful'],
            'graphql': ['graphql'],
            'microservices': ['microservices', 'microservice architecture'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment'],
            'linux': ['linux', 'unix'],
            'bash': ['bash', 'shell scripting'],
            'powershell': ['powershell'],
        }
        
        # Skill relationships for query expansion
        self.skill_relationships = self._build_skill_relationships()
        
    def _load_spacy(self):
        """Load spaCy model with GPU support."""
        try:
            import spacy
            
            # Try to use GPU
            try:
                spacy.require_gpu()
                print("✓ spaCy using GPU")
            except:
                print("⚠ spaCy using CPU (GPU not available)")
            
            # Load smaller model for faster processing
            self.nlp = spacy.load('en_core_web_sm')
            
            # Disable unnecessary pipeline components for speed
            # We only need NER
            self.nlp.select_pipes(enable=['ner'])
            
        except:
            print("⚠ spaCy not available, using regex only")
            self.nlp = None
    
    def _build_skill_relationships(self) -> Dict[str, Set[str]]:
        """Build skill relationship graph for query expansion."""
        relationships = defaultdict(set)
        
        # Programming language ecosystems
        relationships['python'].update(['django', 'flask', 'fastapi', 'pandas', 'numpy', 'machine learning', 'data science'])
        relationships['javascript'].update(['react', 'angular', 'vue', 'nodejs', 'typescript'])
        relationships['typescript'].update(['javascript', 'react', 'angular', 'nodejs'])
        relationships['java'].update(['spring', 'kotlin', 'android'])
        
        # Frontend ecosystem
        relationships['react'].update(['javascript', 'typescript', 'html', 'css', 'nodejs'])
        relationships['angular'].update(['javascript', 'typescript', 'html', 'css'])
        relationships['vue'].update(['javascript', 'html', 'css'])
        
        # Backend ecosystem
        relationships['nodejs'].update(['javascript', 'typescript', 'react', 'mongodb'])
        relationships['django'].update(['python', 'postgresql', 'rest api'])
        relationships['flask'].update(['python', 'rest api'])
        relationships['spring'].update(['java', 'kotlin', 'microservices'])
        
        # Cloud & DevOps
        relationships['aws'].update(['cloud', 'devops', 'docker', 'kubernetes', 'terraform'])
        relationships['azure'].update(['cloud', 'devops', 'docker', 'kubernetes'])
        relationships['gcp'].update(['cloud', 'devops', 'docker', 'kubernetes'])
        relationships['docker'].update(['kubernetes', 'devops', 'ci/cd', 'linux'])
        relationships['kubernetes'].update(['docker', 'devops', 'aws', 'azure', 'gcp'])
        
        # Data Science
        relationships['machine learning'].update(['python', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'data science'])
        relationships['data science'].update(['python', 'machine learning', 'pandas', 'numpy', 'sql'])
        relationships['tensorflow'].update(['python', 'machine learning', 'deep learning'])
        relationships['pytorch'].update(['python', 'machine learning', 'deep learning'])
        
        # Database relationships
        relationships['postgresql'].update(['sql', 'python', 'nodejs'])
        relationships['mongodb'].update(['nodejs', 'python', 'nosql'])
        relationships['mysql'].update(['sql', 'php', 'python'])
        
        return relationships
    
    def extract_skills_batch(self, texts: List[str], batch_size: int = 1000) -> List[Set[str]]:
      """Extract skills from multiple texts efficiently."""
      
      print("Phase 1: Regex pattern matching...")
      all_extracted_skills = []
      
      # Phase 1: Fast regex matching (parallel-friendly)
      for text in tqdm(texts, desc="Regex extraction"):
          if not text or not isinstance(text, str):
              all_extracted_skills.append(set())
              continue
          
          text_lower = text.lower()
          extracted_skills = set()
          
          # Pattern matching (fast)
          for skill_key, variations in self.skill_taxonomy.items():
              for variation in variations:
                  if variation in text_lower:  # Simple substring first
                      # Then verify with word boundary
                      pattern = r'\b' + re.escape(variation) + r'\b'
                      if re.search(pattern, text_lower):
                          extracted_skills.add(skill_key)
                          break
          
          all_extracted_skills.append(extracted_skills)
      
      # Phase 2: spaCy NER (if available and needed)
      if self.nlp:
          print("Phase 2: spaCy NER processing...")
          
          # Filter texts that need NER (optional - saves time)
          texts_for_ner = [t[:10000] for t in texts]  # Limit length for performance
          
          # Process in batches with GPU
          for i, doc in enumerate(tqdm(
              self.nlp.pipe(
                  texts_for_ner,
                  batch_size=batch_size,
                  disable=['parser', 'tagger', 'lemmatizer'],
                  n_process=1  # Use 1 process for GPU
              ),
              total=len(texts_for_ner),
              desc="spaCy NER"
          )):
              for ent in doc.ents:
                  if ent.label_ in ['PRODUCT', 'ORG']:
                      ent_lower = ent.text.lower()
                      for skill_key, variations in self.skill_taxonomy.items():
                          if ent_lower in variations:
                              all_extracted_skills[i].add(skill_key)
                              break
      
      return all_extracted_skills
    def extract_skills(self, text: str) -> Set[str]:
        """Extract skills from text using regex and NER."""
        if not text or not isinstance(text, str):
            return set()
        
        text_lower = text.lower()
        extracted_skills = set()
        
        # Pattern matching for skills
        for skill_key, variations in self.skill_taxonomy.items():
            for variation in variations:
                # Use word boundaries for accurate matching
                pattern = r'\b' + re.escape(variation) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_skills.add(skill_key)
                    break
        
        # Use spaCy NER for additional tech terms
        if self.nlp:
            doc = self.nlp(text[:10000])  # Limit text length for performance
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'ORG']:
                    ent_lower = ent.text.lower()
                    # Check if it matches any skill
                    for skill_key, variations in self.skill_taxonomy.items():
                        if ent_lower in variations:
                            extracted_skills.add(skill_key)
                            break
        
        return extracted_skills
    
    def expand_query_skills(self, skills: Set[str]) -> Set[str]:
        """Expand query skills with related skills."""
        expanded = set(skills)
        for skill in skills:
            if skill in self.skill_relationships:
                # Add top related skills (limit to avoid too much expansion)
                expanded.update(list(self.skill_relationships[skill])[:3])
        return expanded
    
    def get_skill_categories(self) -> Dict[str, List[str]]:
        """Get skills organized by categories for UI."""
        categories = {
            'Programming Languages': ['python', 'javascript', 'java', 'typescript', 'cpp', 'csharp', 'go', 'rust', 'php', 'ruby'],
            'Web Development': ['react', 'angular', 'vue', 'html', 'css', 'nodejs', 'django', 'flask', 'spring'],
            'Databases': ['sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch'],
            'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'Data Science & ML': ['machine learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'data science'],
            'Mobile': ['ios', 'android', 'react native', 'flutter'],
            'Other': ['git', 'agile', 'rest api', 'microservices', 'testing']
        }
        return categories
    
    def get_all_skills(self) -> List[str]:
        """Get all available skills for autocomplete."""
        return sorted(list(self.skill_taxonomy.keys()))
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill name to canonical form."""
        skill_lower = skill.lower().strip()
        for canonical, variations in self.skill_taxonomy.items():
            if skill_lower in variations or skill_lower == canonical:
                return canonical
        return skill_lower