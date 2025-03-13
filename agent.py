"""
Agent module for the Travel Plan Agent.
This module implements the AI agent that generates personalized travel plans.
"""

import time
import google.generativeai as genai
from prompts import TravelPrompts

class TravelAgent:
    """Travel agent class that uses Google Gemini to generate travel plans."""
    
    def __init__(self, config, rag_database):
        """
        Initialize the travel agent.
        
        Args:
            config: Application configuration object
            rag_database: RAG database instance
        """
        self.config = config
        self.rag_db = rag_database
        self.prompts = TravelPrompts()
        
        # Configure Google Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        # Create Gemini model
        self.model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            generation_config={
                "max_output_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "top_k": self.config.TOP_K
            }
        )
    
    def generate_draft_plan(self, user_query: str) -> str:
        """
        Generate a draft travel plan based on user query.
        
        Args:
            user_query: User's travel plan request
            
        Returns:
            Draft travel plan as a string
        """
        print("Generating draft travel plan...")
        
        # Retrieve relevant context from the database
        context = self.rag_db.get_relevant_context(user_query, n_results=3)
        
        # Create prompt for draft plan generation
        prompt = self.prompts.get_draft_plan_prompt(user_query, context)
        
        # Generate draft plan
        start_time = time.time()
        response = self.model.generate_content(prompt)
        end_time = time.time()
        
        print(f"Draft plan generated in {end_time - start_time:.2f} seconds.")
        
        return response.text
    
    def generate_detailed_plan(self, user_query: str, draft_plan: str, feedback: str) -> str:
        """
        Generate a detailed travel plan based on draft plan and user feedback.
        
        Args:
            user_query: Original user query
            draft_plan: Draft travel plan
            feedback: User feedback on the draft plan
            
        Returns:
            Detailed travel plan as a string
        """
        print("Generating detailed travel plan...")
        
        # Retrieve more extensive context for the detailed plan
        context = self.rag_db.get_relevant_context(user_query, n_results=5)
        
        # Create prompt for detailed plan generation
        prompt = self.prompts.get_detailed_plan_prompt(user_query, draft_plan, feedback, context)
        
        # Generate detailed plan
        start_time = time.time()
        response = self.model.generate_content(prompt)
        end_time = time.time()
        
        print(f"Detailed plan generated in {end_time - start_time:.2f} seconds.")
        
        return response.text
    
    def validate_plan(self, plan: str) -> bool:
        """
        Validate the plausibility of a generated travel plan.
        
        Args:
            plan: Travel plan to validate
            
        Returns:
            True if the plan is valid, False otherwise
        """
        # Create prompt for validation
        prompt = self.prompts.get_validation_prompt(plan)
        
        # Generate validation response
        response = self.model.generate_content(prompt)
        
        # Parse validation result
        result = response.text.strip().lower()
        return "valid" in result or "yes" in result