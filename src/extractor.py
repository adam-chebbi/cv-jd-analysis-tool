import pdfplumber
import spacy
import yaml
import logging
import os
from typing import List, Optional

class SkillExtractor:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the SkillExtractor with configuration and Spacy model."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set up logging
        os.makedirs(self.config['paths']['logs'], exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.config['paths']['logs'], 'extractor.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load Spacy model
        try:
            self.nlp = spacy.load(self.config['extractor']['model'])
            self.logger.info(f"Loaded Spacy model: {self.config['extractor']['model']}")
        except Exception as e:
            self.logger.error(f"Failed to load Spacy model: {str(e)}")
            raise
        
        # Define a simple skill list for pattern matching (extendable)
        self.skill_patterns = [
            "python", "java", "javascript", "sql", "project management",
            "communication", "teamwork", "data analysis", "machine learning"
        ]

    def validate_file(self, file_path: str) -> bool:
        """Validate the CV file (PDF format and size)."""
        if not file_path.lower().endswith('.pdf'):
            self.logger.error(f"Invalid file format: {file_path}. Must be PDF.")
            return False
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config['extractor']['max_file_size_mb']:
            self.logger.error(f"File {file_path} exceeds size limit of {self.config['extractor']['max_file_size_mb']} MB")
            return False
        
        return True

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """Extract text from a PDF file."""
        if not self.validate_file(file_path):
            return None
        
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                self.logger.info(f"Successfully extracted text from {file_path}")
                return text
        except Exception as e:
            self.logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            return None

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using NLP and pattern matching."""
        if not text:
            self.logger.warning("No text provided for skill extraction")
            return []
        
        try:
            doc = self.nlp(text.lower())
            skills = set()  # Use set to avoid duplicates
            
            # Simple pattern matching for predefined skills
            for skill in self.skill_patterns:
                if skill in text.lower():
                    skills.add(skill)
            
            # Extract noun phrases that might represent skills
            for chunk in doc.noun_chunks:
                if any(skill in chunk.text.lower() for skill in self.skill_patterns):
                    skills.add(chunk.text.lower())
            
            self.logger.info(f"Extracted skills: {skills}")
            return list(skills)
        except Exception as e:
            self.logger.error(f"Error during skill extraction: {str(e)}")
            return []

    def process_cv(self, cv_path: str) -> List[str]:
        """Process a CV file and return extracted skills."""
        text = self.extract_text_from_pdf(cv_path)
        if text:
            return self.extract_skills(text)
        return []
