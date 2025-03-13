"""
Prompts module for the Travel Plan Agent.
This module contains all the prompts used by the Gemini model.
"""

class TravelPrompts:
    """Class containing all prompts for the Travel Plan Agent."""
    
    def get_draft_plan_prompt(self, user_query: str, context: str) -> str:
        """
        Generate prompt for draft travel plan generation.
        
        Args:
            user_query: User's travel plan request
            context: Relevant context from the RAG database
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are an expert travel planner AI that creates personalized travel plans. 
Your task is to create a draft travel plan based on the user's request.

USER REQUEST:
{user_query}

RELEVANT TRAVEL INFORMATION:
{context}

INSTRUCTIONS:
1. Create a draft travel plan responding to the user's request
2. Use the provided travel information where relevant
3. Include suggested destinations, accommodations, and activities
4. Format the plan as a bulleted overview with the following sections:
   - Destinations
   - Duration (suggested length of trip)
   - Accommodations
   - Key Attractions & Activities
   - Transportation
   - Estimated Budget

Keep the draft plan concise (about 300-400 words) as this is just an initial proposal
that the user will provide feedback on.

DRAFT TRAVEL PLAN:
"""

    def get_detailed_plan_prompt(self, user_query: str, draft_plan: str, feedback: str, context: str) -> str:
        """
        Generate prompt for detailed travel plan generation.
        
        Args:
            user_query: Original user query
            draft_plan: Draft travel plan
            feedback: User feedback on the draft plan
            context: Relevant context from the RAG database
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are an expert travel planner AI that creates detailed, personalized travel plans.
Your task is to create a comprehensive travel plan based on the user's request, your draft plan, 
and their feedback.

ORIGINAL USER REQUEST:
{user_query}

DRAFT TRAVEL PLAN:
{draft_plan}

USER FEEDBACK:
{feedback}

RELEVANT TRAVEL INFORMATION:
{context}

INSTRUCTIONS:
1. Create a detailed travel plan that incorporates the user's feedback
2. Use the provided travel information to enhance the plan with specific details
3. Structure the plan in a professional, easy-to-read format with clear headings
4. Include the following sections:
   - Executive Summary (brief overview)
   - Daily Itinerary (day-by-day breakdown)
   - Detailed Accommodations (with specific options from the travel documents)
   - Activities & Attractions (with descriptions and visiting tips)
   - Transportation Details (between and within destinations)
   - Dining Recommendations
   - Estimated Budget Breakdown
   - Practical Tips (cultural norms, packing suggestions, etc.)

5. Match the style, tone, and level of detail found in professional travel documents
6. Ensure all recommendations are realistic, specific, and based on the provided information
7. The plan should be comprehensive (800-1200 words) and ready for the user to use

DETAILED TRAVEL PLAN:
"""

    def get_validation_prompt(self, plan: str) -> str:
        """
        Generate prompt for travel plan validation.
        
        Args:
            plan: Travel plan to validate
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are an expert travel consultant tasked with validating the plausibility of travel plans.
Please review the following travel plan and determine if it is realistic and plausible.

TRAVEL PLAN:
{plan}

VALIDATION CRITERIA:
1. Geographic consistency (are destinations logically connected?)
2. Realistic timeframes (is enough time allocated for activities and travel?)
3. Seasonal appropriateness (are activities suitable for typical weather conditions?)
4. Logical itinerary flow (does the sequence of activities make sense?)
5. Practical transportation (are transportation methods feasible between locations?)

Please evaluate the plan against these criteria and determine if it is valid.
Respond with VALID if the plan is realistic and plausible, or INVALID followed by a 
brief explanation of the issues if it is not realistic.

VALIDATION RESULT:
"""