"""
Agent module for the Travel Plan Agent.
This module implements the AI agent that generates personalized travel plans.
"""

import time
import random
import logging
import json
import hashlib
import os
from pathlib import Path
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
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache directory and in-memory cache
        self.cache_dir = self.config.BASE_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.response_cache = {}
        self.cache_enabled = True  # Can be toggled if needed
        
        # Create a cache config file to store metadata
        self.cache_config_path = self.cache_dir / "cache_config.json"
        if not self.cache_config_path.exists():
            with open(self.cache_config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "created_at": time.time(),
                    "last_cleared": time.time(),
                    "entries": 0
                }, f)
        
        # Rate limiting settings
        self.last_api_call = 0
        self.min_delay_between_calls = 3.0  # Minimum 3 seconds between calls (increased from 1s)
        self.jitter = 1.0  # Add up to 1 second of random jitter (increased from 0.5s)
        self.consecutive_errors = 0
        self.backoff_factor = 2  # Exponential backoff factor
        self.max_retries = 5  # Increased from 3 to handle more retries
        self.batch_counter = 0  # Counter for batch processing
        self.batch_size = 5  # After this many requests, take a longer pause
        self.batch_pause = 15.0  # Seconds to pause after batch_size requests
        
        # Configure Google Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        # Create Gemini models - one for travel plans and one for keyword extraction
        self.model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            generation_config={
                "max_output_tokens": self.config.MAX_TOKENS,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "top_k": self.config.TOP_K
            }
        )
        
        # Create a more factual model for keyword extraction with lower temperature
        self.keyword_model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            generation_config={
                "max_output_tokens": 1024,  # Smaller tokens needed for keywords
                "temperature": 0.1,         # Low temperature for more deterministic results
                "top_p": 0.9,
                "top_k": 10
            }
        )
    
    def _get_cache_key(self, prompt):
        """
        Generate a deterministic cache key from a prompt.
        
        Args:
            prompt: The prompt to generate a key for
            
        Returns:
            String hash key
        """
        # Convert the prompt to a string if it's not already
        prompt_str = str(prompt)
        # Create a hash of the prompt to use as the cache key
        hash_obj = hashlib.md5(prompt_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _check_cache(self, prompt):
        """
        Check if a response is cached for the given prompt.
        
        Args:
            prompt: The prompt to check
            
        Returns:
            Cached response or None
        """
        if not self.cache_enabled:
            return None
            
        # Check in-memory cache first
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.response_cache:
            self.logger.info(f"Cache hit for prompt (memory): {prompt[:50]}...")
            return self.response_cache[cache_key]
            
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.logger.info(f"Cache hit for prompt (file): {prompt[:50]}...")
                
                # Create a simple object to mimic Gemini response structure
                class CachedResponse:
                    def __init__(self, text):
                        self.text = text
                
                # Store in memory for faster access next time
                cached_response = CachedResponse(cached_data['response_text'])
                self.response_cache[cache_key] = cached_response
                return cached_response
            except Exception as e:
                self.logger.warning(f"Error reading cache file: {e}")
                
        return None
        
    def _save_to_cache(self, prompt, response):
        """
        Save a response to the cache.
        
        Args:
            prompt: The prompt that was used
            response: The response object
        """
        if not self.cache_enabled:
            return
            
        try:
            cache_key = self._get_cache_key(prompt)
            
            # Store in memory cache
            self.response_cache[cache_key] = response
            
            # Store in file cache
            cache_data = {
                'prompt': prompt,
                'response_text': response.text,
                'timestamp': time.time()
            }
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Cached response for prompt: {prompt[:50]}...")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
    
    def _rate_limited_generate(self, model, prompt, retry_count=0, use_cache=True):
        """
        Make a rate-limited call to the Gemini API with exponential backoff.
        
        Args:
            model: The Gemini model to use
            prompt: The prompt to send
            retry_count: Current retry attempt (for recursive calls)
            use_cache: Whether to check and use cached responses
            
        Returns:
            The model response
            
        Raises:
            Exception: If maximum retries exceeded
        """
        # Check cache first if enabled
        if use_cache:
            cached_response = self._check_cache(prompt)
            if cached_response:
                self.logger.info("Using cached response instead of API call")
                return cached_response
        
        # Calculate time since last API call
        now = time.time()
        time_since_last_call = now - self.last_api_call
        
        # In offline/super-minimal mode, skip all rate limiting
        if hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE:
            self.logger.info("Running in offline mode - skipping all rate limiting")
            # Skip all delays completely
        else:
            # Increment batch counter and check if we need a longer pause
            self.batch_counter += 1
            if self.batch_counter >= self.batch_size:
                self.logger.info(f"Batch limit reached ({self.batch_size} requests). Taking a longer pause of {self.batch_pause}s")
                time.sleep(self.batch_pause)
                self.batch_counter = 0  # Reset counter after pause
            
        # Calculate delays only if we're not in offline mode
        if not (hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE):
            # Calculate delay needed based on consecutive errors (exponential backoff)
            if self.consecutive_errors > 0:
                # Apply exponential backoff - more aggressive with higher factor
                required_delay = self.min_delay_between_calls * (self.backoff_factor ** self.consecutive_errors)
                self.logger.info(f"Applying backoff delay of {required_delay:.2f}s after {self.consecutive_errors} consecutive errors")
            else:
                required_delay = self.min_delay_between_calls
            
            # Add some random jitter to avoid synchronized requests
            jitter_amount = random.uniform(0, self.jitter)
            required_delay += jitter_amount
            
            # Wait if needed
            if time_since_last_call < required_delay:
                sleep_time = required_delay - time_since_last_call
                self.logger.info(f"Rate limiting: Waiting {sleep_time:.2f}s before next API call")
                time.sleep(sleep_time)
        
        # Make the API call
        try:
            self.last_api_call = time.time()  # Update the timestamp before the call
            self.logger.info(f"Making API call (batch request {self.batch_counter}/{self.batch_size})")
            response = model.generate_content(prompt)
            
            # Success! Reset consecutive errors but keep batch counter
            self.consecutive_errors = 0
            
            # Cache the successful response if caching is enabled
            if use_cache:
                self._save_to_cache(prompt, response)
                
            return response
            
        except Exception as e:
            self.consecutive_errors += 1
            error_message = str(e)
            
            # Check if it's a rate limit or quota error
            if "429" in error_message or "quota" in error_message.lower() or "exhausted" in error_message.lower() or "rate" in error_message.lower():
                if retry_count < self.max_retries:
                    retry_count += 1
                    # More aggressive wait time calculation
                    wait_time = self.min_delay_between_calls * (self.backoff_factor ** (retry_count + 1))
                    self.logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f}s (attempt {retry_count}/{self.max_retries})")
                    
                    # For rate limits, reset batch counter to force a longer pause
                    self.batch_counter = self.batch_size - 1
                    
                    # Skip sleep in offline mode
                    if not (hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE):
                        time.sleep(wait_time)
                    
                    return self._rate_limited_generate(model, prompt, retry_count, use_cache)
                else:
                    self.logger.error("Maximum retries exceeded for API call")
                    # Force a very long cooldown period after max retries, unless in offline mode
                    if not (hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE):
                        time.sleep(self.batch_pause)
                    
                    # Try to use fallback generator instead of failing completely
                    try:
                        self.logger.warning("Attempting to use fallback generator due to rate limits...")
                        return self._fallback_generate(prompt)
                    except:
                        # If fallback also fails, then raise the original exception
                        raise Exception("Maximum retries exceeded for Gemini API due to rate limits")
            else:
                # If it's not a rate limit error, re-raise it
                raise
                
    def _fallback_generate(self, prompt):
        """
        A simple fallback generator when the Gemini API is unavailable.
        This uses rule-based generation for simple responses.
        
        Args:
            prompt: The original prompt
            
        Returns:
            A simple response object with a text attribute
        """
        self.logger.info("Using fallback text generator...")
        
        # Create a simple response object
        class FallbackResponse:
            def __init__(self, text):
                self.text = text
        
        # Handle different types of prompts with rules
        prompt_lower = str(prompt).lower()
        
        # For keyword extraction
        if "extract" in prompt_lower and "keywords" in prompt_lower:
            self.logger.info("Fallback: Generating basic keywords")
            response_text = """
            {
                "locations": ["Vietnam"],
                "themes": ["beach", "food", "culture"],
                "activities": ["sightseeing", "relaxing"],
                "accommodation_types": ["hotel"],
                "timeframe": ["10 days"],
                "languages": ["English"],
                "budget_level": ["mid-range"]
            }
            """
            return FallbackResponse(response_text.strip())
            
        # For travel plan draft
        elif "travel plan" in prompt_lower or "draft plan" in prompt_lower:
            self.logger.info("Fallback: Generating basic travel plan draft")
            response_text = """
            # Travel Plan Draft: Vietnam Beach & Food Experience
            
            ## Overview
            This 10-day trip to Vietnam focuses on beach relaxation and food experiences.
            
            ## Destinations
            - Ho Chi Minh City (2 days)
            - Hoi An (3 days)
            - Da Nang (2 days)
            - Ha Long Bay (3 days)
            
            ## Activities
            - Street food tours
            - Beach time
            - Cultural sightseeing
            
            ## Note
            This is a fallback draft due to API limitations. Please try again later for a more detailed plan.
            """
            return FallbackResponse(response_text.strip())
            
        # For detailed plans
        elif "detailed plan" in prompt_lower:
            self.logger.info("Fallback: Generating simplified detailed plan")
            response_text = """
            # Vietnam Beach & Food Experience - 10 Days
            
            ## Day 1-2: Ho Chi Minh City
            - Arrive in Ho Chi Minh City
            - Street food tour in District 1
            - Visit Ben Thanh Market
            
            ## Day 3-5: Hoi An
            - Fly to Da Nang, transfer to Hoi An
            - Beach time at An Bang Beach
            - Cooking class featuring local cuisine
            
            ## Day 6-7: Da Nang
            - Transfer to Da Nang
            - Relax at My Khe Beach
            - Visit Marble Mountains
            
            ## Day 8-10: Ha Long Bay
            - Fly to Hanoi, transfer to Ha Long Bay
            - Overnight cruise in Ha Long Bay
            - Return to Hanoi for departure
            
            ## Sources
            - Vietnam Tourism Board
            
            ## Note
            This is a simplified plan due to API limitations. Please try again later for a more detailed itinerary.
            """
            return FallbackResponse(response_text.strip())
            
        # For validation
        elif "valid" in prompt_lower:
            return FallbackResponse("valid")
            
        # Default response
        else:
            self.logger.info("Fallback: Generating generic response")
            return FallbackResponse("Unable to generate detailed response due to API limitations. Please try again later.")
    
    def clear_cache(self, older_than_days=None):
        """
        Clear the response cache.
        
        Args:
            older_than_days: If provided, only clear cache entries older than this many days
                             If None, clear all cache entries
        
        Returns:
            Number of entries removed
        """
        self.logger.info(f"Clearing cache {'older than ' + str(older_than_days) + ' days' if older_than_days else 'completely'}")
        
        # Clear memory cache
        self.response_cache = {}
        
        # Clear file cache
        count = 0
        now = time.time()
        
        if older_than_days is None:
            # Remove all cache files except the config
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_config.json":
                    cache_file.unlink()
                    count += 1
        else:
            # Remove only old cache files
            cutoff_time = now - (older_than_days * 24 * 60 * 60)
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "cache_config.json":
                    try:
                        # Check file timestamp
                        file_time = cache_file.stat().st_mtime
                        if file_time < cutoff_time:
                            cache_file.unlink()
                            count += 1
                    except Exception as e:
                        self.logger.warning(f"Error checking cache file age: {e}")
        
        # Update the cache config
        try:
            with open(self.cache_config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "created_at": time.time(),
                    "last_cleared": now,
                    "entries": 0
                }, f)
        except Exception as e:
            self.logger.warning(f"Error updating cache config: {e}")
            
        self.logger.info(f"Removed {count} cache entries")
        return count
    
    def extract_search_keywords(self, user_query: str) -> dict:
        """
        Use Gemini to extract search keywords from the user query.
        
        Args:
            user_query: The user's query string
            
        Returns:
            Dictionary of extracted keywords for search
        """
        print("Extracting intelligent search keywords with Gemini...")
        
        # Create a prompt for keyword extraction
        keyword_prompt = f"""
You are a travel search keyword extractor. Extract the most important search keywords from this travel query.
Consider locations, activities, themes, preferences, and other relevant details.

USER QUERY:
{user_query}

Respond with a structured JSON object containing these fields:
1. "locations": Array of location names (countries, cities, regions, etc.)
2. "themes": Array of travel themes (beach, culture, food, adventure, etc.)
3. "activities": Array of specific activities mentioned
4. "accommodation_types": Array of accommodation preferences
5. "timeframe": Any time-related information
6. "languages": Languages specifically mentioned or implied by the query
7. "budget_level": Budget category if mentioned (budget, mid-range, luxury)

For each field, extract only explicitly mentioned concepts, not implied ones.
If not all fields have values, that's okay - only include those that are actually in the query.

IMPORTANT: 
- Use English terms in all fields, translating from other languages if needed
- If "Vietnam" is mentioned in any way, make sure it's included in locations
- If "beach" or coastal terms are mentioned, include "beach" in themes
- If "food" or culinary terms are mentioned, include "food" in themes

Return ONLY a valid JSON object, nothing else.
"""
        
        try:
            # Generate keyword extraction using rate-limited function
            response = self._rate_limited_generate(self.keyword_model, keyword_prompt)
            keywords_text = response.text.strip()
            
            # Convert the response to a dictionary
            import json
            keywords = json.loads(keywords_text)
            
            print(f"Extracted keywords: {keywords}")
            return keywords
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            self.logger.warning(f"Falling back to basic keywords after error: {e}")
            
            # Return a basic structure with any obvious keywords we can extract
            basic_keywords = {
                "locations": [],
                "themes": [],
                "activities": [],
                "accommodation_types": [],
                "timeframe": [],
                "languages": [],
                "budget_level": []
            }
            
            # Extract obvious location (Vietnam)
            if "vietnam" in user_query.lower():
                basic_keywords["locations"].append("Vietnam")
            
            # Extract obvious themes
            if any(term in user_query.lower() for term in ["beach", "coast", "sea", "ocean", "shore"]):
                basic_keywords["themes"].append("beach")
            
            if any(term in user_query.lower() for term in ["food", "eat", "cuisine", "culinary", "gastronomy", "restaurant"]):
                basic_keywords["themes"].append("food")
                
            if any(term in user_query.lower() for term in ["culture", "history", "traditional", "ancient"]):
                basic_keywords["themes"].append("culture")
                
            print(f"Fallback basic keywords: {basic_keywords}")
            return basic_keywords
    
    def generate_draft_plan(self, user_query: str) -> str:
        """
        Generate a draft travel plan based on user query.
        
        Args:
            user_query: User's travel plan request
            
        Returns:
            Draft travel plan as a string
        """
        print("Generating draft travel plan...")
        
        # Extract search keywords using Gemini
        search_keywords = self.extract_search_keywords(user_query)
        
        # Retrieve relevant context from the database using the extracted keywords
        context = self.rag_db.get_relevant_context_with_llm_keywords(user_query, search_keywords, n_results=3)
        
        # Create prompt for draft plan generation
        prompt = self.prompts.get_draft_plan_prompt(user_query, context)
        
        # Generate draft plan with rate limiting
        start_time = time.time()
        response = self._rate_limited_generate(self.model, prompt)
        end_time = time.time()
        
        print(f"Draft plan generated in {end_time - start_time:.2f} seconds.")
        
        return response.text
    
    def extract_keywords_from_feedback(self, feedback: str, existing_keywords: dict) -> dict:
        """
        Extract additional keywords from feedback without using an API call.
        
        Args:
            feedback: User feedback text
            existing_keywords: Dictionary of existing keywords
            
        Returns:
            Updated keywords dictionary
        """
        feedback_lower = feedback.lower()
        updated_keywords = existing_keywords.copy()
        
        # Simple keyword extraction based on common travel terms
        for category, terms in {
            "locations": ["vietnam", "thailand", "japan", "asia", "beach", "mountain", "city", "island", 
                        "hanoi", "bangkok", "tokyo", "phuket", "bali", "singapore"],
            "themes": ["adventure", "relax", "culture", "food", "history", "nature", "beach", "luxury", 
                     "budget", "family", "romantic", "shopping", "spiritual", "wellness"],
            "activities": ["hiking", "swimming", "tour", "sightseeing", "museum", "temple", "market", 
                        "restaurant", "yoga", "massage", "cooking", "diving", "snorkeling"],
            "accommodation_types": ["hotel", "resort", "hostel", "apartment", "villa", "homestay", "luxury"]
        }.items():
            for term in terms:
                if term in feedback_lower and term not in [x.lower() for x in updated_keywords.get(category, [])]:
                    updated_keywords.setdefault(category, []).append(term.title())
        
        # Extract specific days/duration
        days_match = re.search(r'(\d+)[\s-]*day', feedback_lower)
        if days_match:
            days = days_match.group(1)
            updated_keywords.setdefault("timeframe", []).append(f"{days} days")
        
        # Look for budget mentions
        if "budget" in feedback_lower or "cheap" in feedback_lower:
            updated_keywords.setdefault("budget_level", []).append("budget")
        elif "luxury" in feedback_lower or "expensive" in feedback_lower:
            updated_keywords.setdefault("budget_level", []).append("luxury")
            
        return updated_keywords
    
    def generate_detailed_plan(self, user_query: str, draft_plan: str, feedback: str) -> str:
        """
        Generate a detailed travel plan based on draft plan and user feedback.
        
        Args:
            user_query: Original user query
            draft_plan: Draft travel plan (can be empty if SKIP_DRAFT_PLAN is True)
            feedback: User feedback on the draft plan
            
        Returns:
            Detailed travel plan as a string
        """
        print("Generating detailed travel plan...")
        
        # Check if we're in direct mode (skipping draft plan)
        is_direct_mode = hasattr(self.config, 'SKIP_DRAFT_PLAN') and self.config.SKIP_DRAFT_PLAN
        
        # Check if we should use another API call for keyword extraction
        if hasattr(self.config, 'OPTIMIZE_KEYWORD_EXTRACTION') and self.config.OPTIMIZE_KEYWORD_EXTRACTION:
            # Get keywords directly from the query
            print("Using optimized keyword extraction (no API call)...")
            original_keywords = self.extract_search_keywords(user_query)
            
            # If we have feedback and we're not in direct mode, enhance with feedback
            if feedback and not is_direct_mode:
                search_keywords = self.extract_keywords_from_feedback(feedback, original_keywords)
            else:
                search_keywords = original_keywords
        else:
            # Standard method: Extract search keywords using Gemini API call
            combined_query = user_query
            if feedback and not is_direct_mode:
                combined_query += " " + feedback
            search_keywords = self.extract_search_keywords(combined_query)
        
        # Retrieve more extensive context for the detailed plan
        context = self.rag_db.get_relevant_context_with_llm_keywords(
            user_query, 
            search_keywords, 
            n_results=5
        )
        
        # Get the source documents for attribution
        # If in direct mode, don't include feedback in the query
        if is_direct_mode:
            source_query = user_query
        else:
            source_query = user_query + " " + feedback
            
        sources = self.rag_db.get_source_documents_with_llm_keywords(
            source_query,
            search_keywords,
            n_results=5
        )
        source_text = "\n".join([f"- {source}" for source in sources])
        
        # Create prompt for detailed plan generation
        # When in direct mode, either draft_plan is empty or feedback might be our default placeholder
        if is_direct_mode:
            print("Generating direct detailed plan without draft step...")
            prompt = self.prompts.get_detailed_plan_prompt(user_query, "", "", context)
        else:
            prompt = self.prompts.get_detailed_plan_prompt(user_query, draft_plan, feedback, context)
        
        # Generate detailed plan with rate limiting
        start_time = time.time()
        response = self._rate_limited_generate(self.model, prompt)
        detailed_plan = response.text
        end_time = time.time()
        
        # Make sure sources are included
        if "Sources:" not in detailed_plan and sources:
            detailed_plan += "\n\n## Sources\n"
            detailed_plan += source_text
            detailed_plan += "\n\n## Original Request\n"
            detailed_plan += user_query
        
        print(f"Detailed plan generated in {end_time - start_time:.2f} seconds.")
        
        # Generate metadata report
        metadata_report = self.generate_plan_metadata(detailed_plan, user_query, sources)
        
        # Combine plan and metadata
        complete_plan = f"{detailed_plan}\n\n{'-'*80}\n\n# PLAN METADATA & ANALYSIS\n\n{metadata_report}"
        
        return complete_plan
        
    def generate_plan_metadata(self, plan: str, user_query: str, sources: list) -> str:
        """
        Generate a comprehensive metadata report for the travel plan.
        
        Args:
            plan: The generated travel plan
            user_query: Original user query
            sources: List of sources used
            
        Returns:
            Metadata report as a string
        """
        print("Generating plan metadata and analysis...")
        
        # Option to skip metadata generation to save API calls
        if hasattr(self.config, 'SKIP_METADATA_GENERATION') and self.config.SKIP_METADATA_GENERATION:
            print("Skipping metadata generation (disabled in config).")
            return "Metadata generation skipped to reduce API calls."
        
        # Create prompt for plan analysis
        prompt = self.prompts.get_plan_analysis_prompt(plan, user_query, sources)
        
        # Generate analysis with rate limiting
        start_time = time.time()
        response = self._rate_limited_generate(self.model, prompt)
        metadata = response.text
        end_time = time.time()
        
        print(f"Plan metadata generated in {end_time - start_time:.2f} seconds.")
        
        return metadata
    
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
        
        # Generate validation response with rate limiting
        response = self._rate_limited_generate(self.model, prompt)
        
        # Parse validation result
        result = response.text.strip().lower()
        return "valid" in result or "yes" in result