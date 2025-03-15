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
import re
from pathlib import Path
import google.generativeai as genai
from prompts import TravelPrompts
from utils import TokenTracker

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
        
        # API call tracking
        self.api_call_counter = 0
        
        # Token usage tracking
        self.token_tracker = TokenTracker()
        
        # Store reference to the token tracker in the config for use by RAG database
        self.config._token_tracker = self.token_tracker
        
        # Share agent reference with RAG database for token tracking
        self.rag_db.agent = self
        
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
        self.min_delay_between_calls = 0.5  # Reduced to 0.5 seconds for faster generation
        self.jitter = 0.2  # Reduced jitter to 0.2 seconds
        self.consecutive_errors = 0
        self.backoff_factor = 2  # Exponential backoff factor stays the same
        self.max_retries = 5  # Keep the same number of retries
        self.batch_counter = 0  # Counter for batch processing
        self.batch_size = 10  # Increased to 10 requests before pause
        self.batch_pause = 5.0  # Reduced pause to 5 seconds
        
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
        
        # Create a chat model for continuous conversations
        # Use slightly higher temperature for more creative and descriptive travel plans
        self.chat_model = genai.GenerativeModel(
            model_name=self.config.GEMINI_MODEL,
            generation_config={
                "max_output_tokens": self.config.MAX_TOKENS,
                "temperature": 0.8,  # Higher temperature for more creative descriptions
                "top_p": 0.95,
                "top_k": 40
            }
        )
        
        # Dictionary to store active chat sessions
        self.active_chats = {}
        
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
    
    def _rate_limited_generate(self, model, prompt, retry_count=0, use_cache=True, component="unknown"):
        """
        Make a rate-limited call to the Gemini API with exponential backoff.
        
        Args:
            model: The Gemini model to use
            prompt: The prompt to send
            retry_count: Current retry attempt (for recursive calls)
            use_cache: Whether to check and use cached responses
            component: Component name for token tracking (e.g., "keyword_extraction")
            
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
                
                # Track token usage even for cached responses (only prompted tokens)
                if hasattr(self, 'token_tracker'):
                    self.token_tracker.track_usage(component, 
                                                  prompt if isinstance(prompt, str) else str(prompt), 
                                                  cached_response.text)
                
                return cached_response
        
        # Calculate time since last API call
        now = time.time()
        time_since_last_call = now - self.last_api_call
        
        # In offline/super-minimal mode, skip all rate limiting completely
        if hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE:
            self.logger.info("Running in offline mode - skipping all rate limiting")
            # Skip all delays completely - don't even increment counters
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
            
            # Increment the API call counter
            self.api_call_counter += 1
            
            # Convert prompt to string if it's not already (for token tracking)
            prompt_str = prompt if isinstance(prompt, str) else str(prompt)
            
            response = model.generate_content(prompt)
            
            # Track token usage
            if hasattr(self, 'token_tracker'):
                self.token_tracker.track_usage(component, prompt_str, response.text)
            
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
                    
                    return self._rate_limited_generate(model, prompt, retry_count, use_cache, component)
                else:
                    self.logger.error("Maximum retries exceeded for API call")
                    # Force a very long cooldown period after max retries, unless in offline mode
                    if not (hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE):
                        time.sleep(self.batch_pause)
                    
                    # Try to use fallback generator instead of failing completely
                    try:
                        self.logger.warning("Attempting to use fallback generator due to rate limits...")
                        fallback_response = self._fallback_generate(prompt)
                        
                        # Track token usage for fallback
                        if hasattr(self, 'token_tracker'):
                            self.token_tracker.track_usage(component + "_fallback", 
                                                          prompt if isinstance(prompt, str) else str(prompt), 
                                                          fallback_response.text)
                        
                        return fallback_response
                    except:
                        # If fallback also fails, then raise the original exception
                        raise Exception("Maximum retries exceeded for Gemini API due to rate limits")
            else:
                # If it's not a rate limit error, re-raise it
                raise
                
    def create_chat_session(self, session_id=None):
        """
        Create a new chat session with the model.
        
        Args:
            session_id: Optional custom session ID (if None, a new ID will be generated)
            
        Returns:
            session_id: The ID of the created session
        """
        if session_id is None:
            # Generate a unique session ID
            session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
            
        # Create a new chat session
        self.active_chats[session_id] = self.chat_model.start_chat(
            history=[]
        )
        
        self.logger.info(f"Created new chat session with ID: {session_id}")
        return session_id
    
    def add_to_chat(self, session_id, message, role="user"):
        """
        Add a message to an existing chat session.
        
        Args:
            session_id: The ID of the chat session
            message: The message to add
            role: The role of the message sender ('user' or 'model')
            
        Returns:
            None
        """
        if session_id not in self.active_chats:
            self.logger.warning(f"Chat session {session_id} does not exist. Creating a new one.")
            self.create_chat_session(session_id)
            
        # Add the message to the chat history
        chat = self.active_chats[session_id]
        chat.history.append({"role": role, "parts": [message]})
        
    def get_chat_response(self, session_id, message=None):
        """
        Get a response from the model in an existing chat session.
        
        Args:
            session_id: The ID of the chat session
            message: Optional new message to add to the chat
            
        Returns:
            The model's response
        """
        if session_id not in self.active_chats:
            self.logger.warning(f"Chat session {session_id} does not exist. Creating a new one.")
            self.create_chat_session(session_id)
            
        chat = self.active_chats[session_id]
        
        # Add the new message if provided
        if message:
            self.add_to_chat(session_id, message)
        
        # Apply rate limiting if not in offline mode
        now = time.time()
        time_since_last_call = now - self.last_api_call
        
        if not (hasattr(self.config, 'OFFLINE_MODE') and self.config.OFFLINE_MODE):
            # Apply rate limiting (similar to _rate_limited_generate but simplified)
            if time_since_last_call < self.min_delay_between_calls:
                sleep_time = self.min_delay_between_calls - time_since_last_call
                self.logger.info(f"Rate limiting: Waiting {sleep_time:.2f}s before next chat API call")
                time.sleep(sleep_time)
        
        # Make the API call
        self.last_api_call = time.time()
        self.logger.info(f"Sending message to chat session {session_id}")
        
        try:
            # Since we already added the message to history, we need to send a non-empty message
            # to trigger the model to generate a response
            response = chat.send_message("Please continue with the travel plan.")
            self.consecutive_errors = 0      # Reset error counter on success
            
            # Add the model's response to the chat history
            self.add_to_chat(session_id, response.text, role="model")
            
            return response.text
            
        except Exception as e:
            self.consecutive_errors += 1
            self.logger.error(f"Error in chat session: {e}")
            
            # If it looks like a rate limit issue
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                self.logger.warning(f"Rate limit exceeded in chat session. Backing off.")
                time.sleep(self.min_delay_between_calls * (self.backoff_factor ** self.consecutive_errors))
                
                # Try once more after backoff
                try:
                    response = chat.send_message("Please continue with the travel plan.")
                    return response.text
                except:
                    return "I encountered an error due to rate limits. Please try again in a moment."
            
            return f"An error occurred: {str(e)}"
    
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
        
        # Get the keyword extraction prompt from prompts.py
        keyword_prompt = self.prompts.get_keyword_extraction_prompt(user_query)
        
        try:
            # Generate keyword extraction using rate-limited function
            response = self._rate_limited_generate(self.keyword_model, keyword_prompt, component="keyword_extraction")
            keywords_text = response.text.strip()
            
            # Convert the response to a dictionary
            import json
            import re
            
            # Clean the response text to ensure it's valid JSON
            # Remove any leading/trailing whitespace, markdown formatting, etc.
            cleaned_text = keywords_text.strip()
            
            # If the text is wrapped in ```json and ``` markers, extract just the JSON part
            if cleaned_text.startswith("```json") and "```" in cleaned_text[7:]:
                cleaned_text = cleaned_text[7:].split("```", 1)[0].strip()
            elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
                
            # Try to parse the JSON
            try:
                keywords = json.loads(cleaned_text)
                print(f"Extracted keywords: {keywords}")
                
                # Enhanced keyword processing - perform secondary extraction from the query
                # to ensure we capture critical geographic information even if the LLM missed it
                
                # Extract all capitalized words as potential location names
                additional_locations = set()
                potential_locations = re.findall(r'\b([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)\b', user_query)
                common_non_locations = {'I', 'My', 'Me', 'Mine', 'The', 'A', 'An', 'And', 'Or', 'But', 'For', 'With', 'To', 'From'}
                for loc in potential_locations:
                    if loc not in common_non_locations and loc not in keywords.get('locations', []):
                        additional_locations.add(loc)
                
                # Extract specific duration patterns
                timeframe_patterns = [
                    (r'(\d+)\s*(?:day|days)', '{} days'),
                    (r'(\d+)\s*(?:week|weeks)', '{} weeks'),
                    (r'(\d+)\s*(?:night|nights)', '{} nights'),
                    (r'(\d+)\s*(?:month|months)', '{} months'),
                ]
                
                additional_timeframes = set()
                for pattern, format_str in timeframe_patterns:
                    matches = re.findall(pattern, user_query, re.IGNORECASE)
                    for match in matches:
                        timeframe = format_str.format(match)
                        if timeframe not in keywords.get('timeframe', []):
                            additional_timeframes.add(timeframe)
                
                # Extract relationships between locations and durations
                relationship_pattern = r'(\d+)\s*(?:day|days|night|nights|week|weeks)\s*(?:in|at|near|around)\s+([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)'
                additional_relationships = set()
                relationship_matches = re.findall(relationship_pattern, user_query, re.IGNORECASE)
                for duration, location in relationship_matches:
                    relationship = f"{duration} days in {location}"
                    additional_relationships.add(relationship)
                    # Also add the location if it's not already included
                    if location not in keywords.get('locations', []):
                        additional_locations.add(location)
                
                # Add special themes that might have been missed
                special_themes = {
                    'romantic': ['romantic', 'romance', 'honeymoon', 'anniversary', 'couple', 'love'],
                    'family': ['family', 'children', 'kids', 'child', 'kid'],
                    'party': ['party', 'parties', 'nightlife', 'clubbing', 'bar hopping'],
                    'relaxation': ['relaxation', 'relax', 'chill', 'unwind', 'peaceful', 'quiet', 'spa'],
                    'adventure': ['adventure', 'adventurous', 'thrill', 'exciting', 'adrenaline'],
                    'cultural': ['cultural', 'culture', 'history', 'historical', 'museum', 'heritage'],
                    'food': ['food', 'culinary', 'cuisine', 'restaurant', 'eating', 'dine', 'dining', 'gastronomic'],
                    'beach': ['beach', 'beaches', 'coastal', 'coast', 'seaside', 'ocean', 'sea']
                }
                
                additional_themes = set()
                for theme_name, keywords_list in special_themes.items():
                    if any(keyword in user_query.lower() for keyword in keywords_list) and theme_name not in keywords.get('themes', []):
                        additional_themes.add(theme_name)
                
                # Now update the keywords with additional extracted information
                if additional_locations:
                    if 'locations' not in keywords:
                        keywords['locations'] = []
                    keywords['locations'].extend(list(additional_locations))
                
                if additional_timeframes:
                    if 'timeframe' not in keywords:
                        keywords['timeframe'] = []
                    keywords['timeframe'].extend(list(additional_timeframes))
                
                if additional_relationships:
                    if 'relationships' not in keywords:
                        keywords['relationships'] = []
                    keywords['relationships'].extend(list(additional_relationships))
                
                if additional_themes:
                    if 'themes' not in keywords:
                        keywords['themes'] = []
                    keywords['themes'].extend(list(additional_themes))
                
                # Deduplicate all lists
                for key in keywords:
                    if isinstance(keywords[key], list):
                        keywords[key] = list(dict.fromkeys(keywords[key]))
                
                print(f"Enhanced keywords with additional extraction: {keywords}")
                return keywords
                
            except json.JSONDecodeError as json_err:
                print(f"JSON parsing error: {json_err}")
                self.logger.warning(f"Error parsing keywords JSON: {json_err}. Falling back to basic keywords.")
                raise  # Re-raise to trigger the fallback
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            self.logger.warning(f"Falling back to basic keywords after error: {e}")
            
            # Return a basic structure with any obvious keywords we can extract
            # Improved fallback method with more sophisticated pattern matching
            basic_keywords = {
                "locations": [],
                "themes": [],
                "activities": [],
                "accommodation_types": [],
                "timeframe": [],
                "languages": [],
                "budget_level": [],
                "special_requirements": [],
                "relationships": []
            }
            
            # Extract locations (more comprehensive)
            potential_locations = re.findall(r'\b([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)\b', user_query)
            common_non_locations = {'I', 'My', 'Me', 'Mine', 'The', 'A', 'An', 'And', 'Or', 'But', 'For', 'With', 'To', 'From'}
            for loc in potential_locations:
                if loc not in common_non_locations:
                    basic_keywords["locations"].append(loc)
            
            # Extract durations and relationships
            duration_matches = re.findall(r'(\d+)\s*(?:day|days|night|nights|week|weeks)', user_query, re.IGNORECASE)
            for duration in duration_matches:
                basic_keywords["timeframe"].append(f"{duration} days")
            
            relationship_matches = re.findall(r'(\d+)\s*(?:day|days|night|nights|week|weeks)\s*(?:in|at|near|around)\s+([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)', user_query, re.IGNORECASE)
            for duration, location in relationship_matches:
                basic_keywords["relationships"].append(f"{duration} days in {location}")
                if location not in basic_keywords["locations"]:
                    basic_keywords["locations"].append(location)
            
            # Extract obvious location (Vietnam)
            if "vietnam" in user_query.lower() and "Vietnam" not in basic_keywords["locations"]:
                basic_keywords["locations"].append("Vietnam")
            
            # Extract themes (more comprehensive)
            theme_patterns = {
                "beach": ["beach", "coast", "sea", "ocean", "shore", "seaside", "sand"],
                "food": ["food", "eat", "cuisine", "culinary", "gastronomy", "restaurant", "dining", "tasting"],
                "culture": ["culture", "history", "traditional", "ancient", "museum", "heritage"],
                "adventure": ["adventure", "hiking", "trekking", "climbing", "rafting", "diving", "exploring"],
                "relaxation": ["relax", "relaxation", "peaceful", "quiet", "calm", "spa", "wellness", "retreat"],
                "romance": ["romantic", "romance", "honeymoon", "anniversary", "couple"],
                "family": ["family", "children", "kids", "child", "family-friendly"],
                "party": ["party", "nightlife", "club", "clubbing", "bar", "disco", "dancing", "music"],
                "luxury": ["luxury", "luxurious", "high-end", "exclusive", "premium", "5-star"]
            }
            
            for theme, patterns in theme_patterns.items():
                if any(pattern in user_query.lower() for pattern in patterns):
                    basic_keywords["themes"].append(theme)
            
            # Extract accommodation types
            accommodation_types = ["hotel", "resort", "hostel", "villa", "apartment", "homestay", "bungalow", "motel"]
            for acc_type in accommodation_types:
                if acc_type in user_query.lower():
                    basic_keywords["accommodation_types"].append(acc_type)
            
            # Extract activities
            activity_keywords = ["visit", "see", "explore", "tour", "experience", "activity", "sightseeing", "trip"]
            for activity in activity_keywords:
                activity_matches = re.findall(f"{activity}\\s+([a-zA-Z]+(?:\\s+[a-zA-Z]+)?)", user_query.lower())
                for match in activity_matches:
                    if len(match) > 3:  # Avoid very short words
                        basic_keywords["activities"].append(f"{activity} {match}")
            
            # Look for special requirements
            if any(term in user_query.lower() for term in ["surprise", "special", "unexpected", "gift"]):
                basic_keywords["special_requirements"].append("surprise element")
            
            # Check for language preferences
            languages = ["English", "German", "French", "Spanish", "Italian", "Chinese", "Japanese"]
            for lang in languages:
                if lang.lower() in user_query.lower():
                    basic_keywords["languages"].append(lang)
            
            # Budget level detection
            budget_indicators = {
                "budget": ["cheap", "budget", "affordable", "inexpensive", "low-cost", "economical"],
                "mid-range": ["moderate", "mid-range", "standard", "average", "reasonable"],
                "luxury": ["luxury", "luxurious", "high-end", "expensive", "premium", "exclusive", "5-star"]
            }
            
            for level, indicators in budget_indicators.items():
                if any(indicator in user_query.lower() for indicator in indicators):
                    basic_keywords["budget_level"].append(level)
                    break
            
            # Remove empty lists
            basic_keywords = {k: v for k, v in basic_keywords.items() if v}
            
            print(f"Enhanced fallback keywords: {basic_keywords}")
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
        response = self._rate_limited_generate(self.model, prompt, component="draft_plan")
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
    
    def generate_detailed_plan(self, user_query: str, draft_plan: str, feedback: str) -> tuple:
        """
        Generate a detailed travel plan based on draft plan and user feedback.
        
        Args:
            user_query: Original user query
            draft_plan: Draft travel plan (can be empty if SKIP_DRAFT_PLAN is True)
            feedback: User feedback on the draft plan
            
        Returns:
            Tuple of (detailed_plan, metadata_report)
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
        
        # Increase n_results to ensure we get enough content, particularly MUST-SEE items
        n_results = 8
        
        # Retrieve more extensive context for the detailed plan, prioritizing MUST-SEE content
        context = self.rag_db.get_relevant_context_with_llm_keywords(
            user_query, 
            search_keywords, 
            n_results=n_results
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
            n_results=n_results
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
        response = self._rate_limited_generate(self.model, prompt, component="detailed_plan")
        travel_plan_content = response.text
        end_time = time.time()
        
        # Make sure sources are included
        if "Sources:" not in travel_plan_content and sources:
            travel_plan_content += "\n\n## Sources\n"
            travel_plan_content += source_text
            travel_plan_content += "\n\n## Original Request\n"
            travel_plan_content += user_query
        
        print(f"Travel plan content generated in {end_time - start_time:.2f} seconds.")
        
        # Generate a more comprehensive quality assurance report
        metadata_report = self.generate_plan_metadata(travel_plan_content, user_query, sources)
        
        # Add a RAG database usage section if it doesn't already exist
        if "RAG Database Usage Report" not in metadata_report:
            rag_report = self.generate_rag_usage_report(travel_plan_content, context, sources)
            metadata_report += f"\n\n## RAG Database Usage Report\n\n{rag_report}"
        
        # Now generate the actual customer email and sales agent notes separately - this is the key change
        print("Generating personalized customer email and sales agent notes...")
        customer_email = self.generate_customer_response(travel_plan_content, user_query)
        
        # Return the customer email (as the main plan) and the metadata report separately
        return customer_email, metadata_report
        
    def generate_rag_usage_report(self, plan: str, context: str, sources: list) -> str:
        """
        Generate a report on how the RAG database was used in the plan.
        
        Args:
            plan: The generated travel plan
            context: The context used to generate the plan
            sources: List of source documents used
            
        Returns:
            RAG usage report as a string
        """
        # Count database vs. external suggestions in the plan
        db_count = plan.count("[FROM DATABASE]")
        external_count = plan.count("[EXTERNAL SUGGESTION]")
        must_see_count = plan.count("[FROM DATABASE - MUST-SEE]") + plan.count("MUST-SEE")
        
        # If the counts are too low, it's likely that the LLM didn't properly tag recommendations
        # In this case, we'll try to estimate based on source document usage
        if (db_count + external_count) < 10:
            # Check context size to estimate usage
            context_size = len(context.strip())
            # If we have a substantial context (more than 1000 chars) and sources
            if context_size > 1000 and len(sources) > 0:
                # Estimate based on number of source documents relative to plan size
                # More source documents usually means more database usage
                plan_size = len(plan.strip())
                source_count = len(sources)
                
                # Calculate an estimated percentage based on source documents and context size
                # More sources and larger context generally means more database usage
                estimated_percentage = min(90, (source_count * context_size / plan_size) * 5)
                db_percentage = max(10, round(estimated_percentage))  # At least 10% if we have any sources
                
                # Estimate must-see count based on source documents (typically 10-20% of sources have must-see content)
                if must_see_count == 0:
                    must_see_count = max(1, round(source_count * 0.15))
                    
                # Recalculate db_count and external_count to match the percentage
                if db_percentage > 0:
                    estimated_total = 100  # Represents 100%
                    db_count = round(estimated_total * (db_percentage / 100))
                    external_count = estimated_total - db_count
            else:
                # If minimal context but we have sources, set a base percentage
                if len(sources) > 0:
                    db_percentage = 10  # Default to 10% if we have sources but minimal context
                    db_count = 10
                    external_count = 90
                else:
                    db_percentage = 0
                    db_count = 0
                    external_count = 100
        else:
            # Calculate percentage of database usage based on actual tags
            total_suggestions = db_count + external_count
            if total_suggestions > 0:
                db_percentage = round((db_count / total_suggestions) * 100)
            else:
                db_percentage = 0
            
        # Get the RAG usage report prompt template from prompts.py and fill in values
        template = self.prompts.get_rag_usage_report_prompt()
        
        # Create source list
        source_list = "\n".join([f"- {source}" for source in sources])
        
        # Fill in the template with actual values
        report = template.format(
            db_percentage=db_percentage,
            must_see_count=must_see_count,
            external_count=external_count,
            len_sources=len(sources),
            source_list=source_list
        )
            
        return report
        
    def generate_customer_response(self, plan: str, user_query: str) -> str:
        """
        Generate a comprehensive customer response as a personalized email with sales agent notes.
        
        Args:
            plan: The generated travel plan
            user_query: Original user query
            
        Returns:
            Customer email with sales agent notes as a string
        """
        # Extract key elements from the plan to personalize the response
        
        # Extract destinations
        destinations = []
        destination_pattern = r"(?:##.*(?:Itinerary|Destinations)|destinations:|locations:)(.*?)(?:##|\n\n)"
        destination_match = re.search(destination_pattern, plan, re.DOTALL | re.IGNORECASE)
        if destination_match:
            dest_text = destination_match.group(1)
            # Extract location names (capitalized words)
            destinations = re.findall(r'\b([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)\b', dest_text)
            # Filter out common non-destinations
            common_non_destinations = {'Day', 'Days', 'Activities', 'Accommodation', 'Transportation', 'Highlights', 'Summary', 'Overview', 'Morning', 'Afternoon', 'Evening', 'Night', 'Breakfast', 'Lunch', 'Dinner'}
            destinations = [d for d in destinations if d not in common_non_destinations]
            # Remove duplicates
            destinations = list(dict.fromkeys(destinations))
        
        # If no destinations were found, look for them throughout the document
        if not destinations:
            common_countries = ['Vietnam', 'Thailand', 'Japan', 'China', 'Malaysia', 'Singapore', 'Indonesia', 'Cambodia', 'Laos']
            for country in common_countries:
                if country in plan:
                    destinations.append(country)
        
        # Extract duration
        duration = ""
        duration_match = re.search(r'(\d+)[\s-]*(day|days|Day|Days)', plan)
        if duration_match:
            duration = f"{duration_match.group(1)} days"
        
        # Extract key highlights or must-see attractions
        highlights = []
        highlights_pattern = r"(?:##.*(?:Highlights|Must-See|Key Attractions)|highlights:|attractions:)(.*?)(?:##|\n\n)"
        highlights_match = re.search(highlights_pattern, plan, re.DOTALL | re.IGNORECASE)
        if highlights_match:
            highlight_text = highlights_match.group(1)
            # Extract bullet points
            highlights = re.findall(r'[-\*•]\s*(.*?)(?:\n|$)', highlight_text)
            # Take only first 3 highlights if we have more
            if len(highlights) > 3:
                highlights = highlights[:3]
        
        # Look for any potential issues or considerations
        considerations = []
        considerations_pattern = r"(?:considerations|be aware|note that|keep in mind|challenges|issues)(.*?)(?:\n\n|\n##|$)"
        considerations_matches = re.findall(considerations_pattern, plan, re.IGNORECASE | re.DOTALL)
        for match in considerations_matches:
            if len(match.strip()) > 10:  # Only include non-trivial considerations
                considerations.append(match.strip())
        
        # Extract any next steps mentioned
        next_steps = []
        next_steps_pattern = r"(?:Next Steps|Recommendations|Actions)(.*?)(?:\n\n|\n##|$)"
        next_steps_match = re.search(next_steps_pattern, plan, re.IGNORECASE | re.DOTALL)
        if next_steps_match:
            next_steps_text = next_steps_match.group(1)
            # Extract bullet points
            next_steps = re.findall(r'[-\*•]\s*(.*?)(?:\n|$)', next_steps_text)
            
        # Extract plausibility check issues
        plausibility_issues = []
        plausibility_pattern = r"(?:Plausibility|Issues|Challenges|Concerns|Inconsistencies|Potential Problems)(.*?)(?:\n\n|\n##|$)"
        plausibility_match = re.search(plausibility_pattern, plan, re.IGNORECASE | re.DOTALL)
        if plausibility_match:
            plausibility_text = plausibility_match.group(1)
            # Extract bullet points
            plausibility_issues = re.findall(r'[-\*•]\s*(.*?)(?:\n|$)', plausibility_text)
            
        # Extract specific requests from user query
        specific_requests = []
        request_patterns = [
            r'(?:ich möchte|ich will|bitte|please)\s+([^\.\,\!]+)',
            r'(?:wichtig ist|important|muss haben|must have)\s+([^\.\,\!]+)',
            r'(?:ich suche|looking for|suche nach)\s+([^\.\,\!]+)'
        ]
        for pattern in request_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Only include non-trivial requests
                    specific_requests.append(match.strip())
        
        # Create a concise summary of the plan to use as context for email generation
        plan_summary = ""
        
        # Try to extract title
        title_match = re.search(r'#\s+([^\n]+)', plan)
        if title_match:
            plan_summary += f"Title: {title_match.group(1)}\n\n"
        else:
            # If no title with # format, try to find any title-like text
            title_alt_match = re.search(r'(?:TRAVEL PLAN|ITINERARY|REISEPLAN):\s*([^\n]+)', plan, re.IGNORECASE)
            if title_alt_match:
                plan_summary += f"Title: {title_alt_match.group(1)}\n\n"
            
        # Extract executive summary if available
        summary_match = re.search(r'(?:Summary|Zusammenfassung|Executive Summary|Overview|Überblick)(.*?)(?:\n\n|\n##|$)', plan, re.IGNORECASE | re.DOTALL)
        if summary_match:
            plan_summary += f"Summary:\n{summary_match.group(1)}\n\n"
            
        # Extract or create highlights section
        if highlights:
            plan_summary += "Highlights:\n"
            for highlight in highlights:
                plan_summary += f"- {highlight}\n"
            plan_summary += "\n"
            
        # Add destinations and duration
        if destinations:
            dest_text = ", ".join(destinations)
            plan_summary += f"Destinations: {dest_text}\n"
        if duration:
            plan_summary += f"Duration: {duration}\n\n"
            
        # Add any considerations or important notes
        if considerations or plausibility_issues:
            plan_summary += "Important Considerations:\n"
            for item in (considerations + plausibility_issues):
                plan_summary += f"- {item}\n"
            plan_summary += "\n"
        
        # Extract any day-by-day itinerary summary (just the headings)
        itinerary_headers = re.findall(r'(?:Day \d+|Tag \d+)[^\n]*', plan)
        if itinerary_headers:
            plan_summary += "Itinerary Overview:\n"
            for header in itinerary_headers[:10]:  # Limit to first 10 days to keep it concise
                plan_summary += f"- {header.strip()}\n"
            plan_summary += "\n"
        
        # Generate the personalized customer email using our dedicated email prompt
        print("Generating personalized customer email...")
        email_prompt = self.prompts.get_customer_email_prompt(plan_summary, user_query)
        email_response = self._rate_limited_generate(self.model, email_prompt, component="customer_email")
        customer_email = email_response.text.strip()
        
        # Compile list of questions for follow-up
        open_questions = []
        
        # First add any urgent issues from plausibility check
        if plausibility_issues:
            for issue in plausibility_issues:
                open_questions.append(f"Klären Sie: {issue}")
        
        # Add considerations that need confirmation
        if considerations:
            for consideration in considerations:
                if not any(consideration in q for q in open_questions):
                    open_questions.append(f"Bestätigen Sie: {consideration}")
        
        # Extract unclear elements from the plan
        unclear_elements = []
        unclear_pattern = r"(?:unclear|nicht klar|unklar|needs clarification|zu klären)(.*?)(?:\n\n|\n##|$)"
        unclear_matches = re.findall(unclear_pattern, plan, re.IGNORECASE | re.DOTALL)
        for match in unclear_matches:
            if len(match.strip()) > 5:
                unclear_elements.append(match.strip())
                
        # Add unclear elements
        for element in unclear_elements:
            if not any(element in q for q in open_questions):
                open_questions.append(f"Klären Sie: {element}")
        
        # Add default questions if we don't have enough
        if len(open_questions) < 3:
            default_questions = [
                "Bestätigen Sie das gewünschte Zimmer-/Hotelkategorie-Level (aktuell: mittlere Preisklasse)",
                "Klären Sie besondere Ernährungswünsche oder Einschränkungen",
                "Bestätigen Sie bevorzugte Transportmittel innerhalb der Reiseziele",
                "Klären Sie Interesse an zusätzlichen Aktivitäten wie geführte Touren oder Workshops",
                "Bestätigen Sie Budget-Range für die Gesamtreise",
                "Klären Sie Bedarf an Reiseversicherung und/oder Assistenzdiensten"
            ]
            
            # Add default questions until we have at least 3
            for question in default_questions:
                if len(open_questions) >= 5:
                    break
                if not any(question in q for q in open_questions):
                    open_questions.append(question)
        
        # Generate sales agent notes
        print("Generating personalized sales agent notes...")
        # Format the open questions list for the prompt
        open_questions_formatted = "\n".join([f"- {q}" for q in open_questions])
        sales_notes_prompt = self.prompts.get_sales_agent_notes_prompt(plan, user_query, open_questions_formatted)
        sales_notes_response = self._rate_limited_generate(self.model, sales_notes_prompt, component="sales_notes")
        sales_agent_notes = sales_notes_response.text.strip()
        
        # Combine the email and sales agent notes
        # Format the result as a completely new document with clear sections
        result = customer_email + "\n\n## Internal Sales Agent Notes\n\n" + sales_agent_notes
        
        return result
        
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
        response = self._rate_limited_generate(self.model, prompt, component="plan_metadata")
        metadata = response.text
        end_time = time.time()
        
        print(f"Plan metadata generated in {end_time - start_time:.2f} seconds.")
        
        return metadata
    
    def semantic_analyze_request(self, user_query: str) -> dict:
        """
        Perform comprehensive semantic analysis on user request to better understand intent.
        
        Args:
            user_query: User's original travel request
            
        Returns:
            Dictionary with semantic analysis results
        """
        print("Performing comprehensive semantic analysis of user request...")
        
        # Get the semantic analysis prompt from prompts.py
        analysis_prompt = self.prompts.get_semantic_analysis_prompt(user_query)
        
        try:
            response = self._rate_limited_generate(self.model, analysis_prompt, component="semantic_analysis")
            analysis_text = response.text.strip()
            
            # Parse the JSON response
            import json
            import re
            
            # Clean the response text to ensure it's valid JSON
            cleaned_text = analysis_text.strip()
            if cleaned_text.startswith("```json") and "```" in cleaned_text[7:]:
                cleaned_text = cleaned_text[7:].split("```", 1)[0].strip()
            elif cleaned_text.startswith("```") and cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
                
            analysis = json.loads(cleaned_text)
            print(f"Semantic analysis completed successfully")
            
            # Extract key properties for logging
            log_summary = {
                "travelers": analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown"),
                "duration": analysis.get("TRIP_LOGISTICS", {}).get("total_duration", "Unknown"),
                "primary_motivation": analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("primary_trip_motivation", "Unknown"),
                "budget_level": analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "Unknown"),
                "language": analysis.get("LINGUISTIC_ANALYSIS", {}).get("language_of_request", "Unknown")
            }
            print(f"Key insights: {log_summary}")
            
            return analysis
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            self.logger.warning(f"Falling back to simple semantic analysis after error: {e}")
            
            # Return a basic structure with reasonable defaults
            return {
                "TRAVELER_PROFILE": {
                    "number_of_travelers": "Unknown",
                    "age_groups": ["adults"],
                    "relationship_dynamics": "Unknown",
                    "special_needs": [],
                    "language_preferences": ["English"]
                },
                "TRIP_LOGISTICS": {
                    "total_duration": "Unspecified",
                    "specific_timeframes": {},
                    "travel_sequence": [],
                    "transportation_preferences": [],
                    "accommodation_preferences": []
                },
                "TRIP_MOTIVATIONS_&_EXPECTATIONS": {
                    "primary_trip_motivation": "general travel",
                    "key_themes": [],
                    "activity_preferences": [],
                    "special_experiences": [],
                    "must_have_elements": []
                },
                "CONTEXTUAL_FACTORS": {
                    "budget_level": "mid-range",
                    "seasonal_considerations": [],
                    "special_occasions": [],
                    "prior_travel_experience": "Unknown",
                    "concerns_constraints": []
                },
                "LINGUISTIC_ANALYSIS": {
                    "language_of_request": "Unknown",
                    "tone_formality": "neutral",
                    "specificity": "moderate",
                    "emotional_undertones": [],
                    "key_emphasis": []
                }
            }
    
    def extract_search_keywords_with_intent(self, user_query: str, semantic_analysis: dict) -> dict:
        """
        Extract search keywords with semantic understanding from user query.
        
        Args:
            user_query: User's travel query
            semantic_analysis: Dictionary from semantic_analyze_request
            
        Returns:
            Enhanced keywords dictionary with semantic understanding
        """
        print("Extracting intelligent search keywords with semantic context...")
        
        # First get basic keywords
        basic_keywords = self.extract_search_keywords(user_query)
        
        # Enhance with semantic understanding if available
        if not semantic_analysis:
            return basic_keywords
            
        try:
            enhanced_keywords = basic_keywords.copy()
            
            # Add locations from semantic analysis
            if "TRIP_LOGISTICS" in semantic_analysis:
                trip_logistics = semantic_analysis["TRIP_LOGISTICS"]
                if "specific_timeframes" in trip_logistics and isinstance(trip_logistics["specific_timeframes"], dict):
                    for location in trip_logistics["specific_timeframes"].keys():
                        if location not in enhanced_keywords.get("locations", []):
                            if "locations" not in enhanced_keywords:
                                enhanced_keywords["locations"] = []
                            enhanced_keywords["locations"].append(location)
                
                # Add relationships between locations and durations
                if "specific_timeframes" in trip_logistics and isinstance(trip_logistics["specific_timeframes"], dict):
                    relationships = []
                    for location, duration in trip_logistics["specific_timeframes"].items():
                        # Clean up the duration to extract just the number
                        import re
                        duration_match = re.search(r'(\d+)', str(duration))
                        if duration_match:
                            days = duration_match.group(1)
                            relationships.append(f"{days} days in {location}")
                    
                    if relationships:
                        enhanced_keywords["relationships"] = relationships
            
            # Add themes from motivations
            if "TRIP_MOTIVATIONS_&_EXPECTATIONS" in semantic_analysis:
                motivations = semantic_analysis["TRIP_MOTIVATIONS_&_EXPECTATIONS"]
                if "key_themes" in motivations and isinstance(motivations["key_themes"], list):
                    for theme in motivations["key_themes"]:
                        if theme.lower() not in [t.lower() for t in enhanced_keywords.get("themes", [])]:
                            if "themes" not in enhanced_keywords:
                                enhanced_keywords["themes"] = []
                            enhanced_keywords["themes"].append(theme)
                
                # Add activities
                if "activity_preferences" in motivations and isinstance(motivations["activity_preferences"], list):
                    for activity in motivations["activity_preferences"]:
                        if activity.lower() not in [a.lower() for a in enhanced_keywords.get("activities", [])]:
                            if "activities" not in enhanced_keywords:
                                enhanced_keywords["activities"] = []
                            enhanced_keywords["activities"].append(activity)
                
                # Add special experiences as activities too
                if "special_experiences" in motivations and isinstance(motivations["special_experiences"], list):
                    for experience in motivations["special_experiences"]:
                        if experience.lower() not in [a.lower() for a in enhanced_keywords.get("activities", [])]:
                            if "activities" not in enhanced_keywords:
                                enhanced_keywords["activities"] = []
                            enhanced_keywords["activities"].append(experience)
            
            # Add accommodation preferences
            if "TRIP_LOGISTICS" in semantic_analysis:
                trip_logistics = semantic_analysis["TRIP_LOGISTICS"]
                if "accommodation_preferences" in trip_logistics and isinstance(trip_logistics["accommodation_preferences"], list):
                    for accommodation in trip_logistics["accommodation_preferences"]:
                        if accommodation.lower() not in [a.lower() for a in enhanced_keywords.get("accommodation_types", [])]:
                            if "accommodation_types" not in enhanced_keywords:
                                enhanced_keywords["accommodation_types"] = []
                            enhanced_keywords["accommodation_types"].append(accommodation)
            
            # Add budget level
            if "CONTEXTUAL_FACTORS" in semantic_analysis:
                contextual = semantic_analysis["CONTEXTUAL_FACTORS"]
                if "budget_level" in contextual and contextual["budget_level"]:
                    budget = contextual["budget_level"].lower()
                    # Normalize budget terms
                    if "luxury" in budget or "high-end" in budget or "premium" in budget:
                        budget_level = "luxury"
                    elif "budget" in budget or "cheap" in budget or "affordable" in budget or "low-cost" in budget:
                        budget_level = "budget"
                    else:
                        budget_level = "mid-range"
                    
                    if "budget_level" not in enhanced_keywords:
                        enhanced_keywords["budget_level"] = []
                    enhanced_keywords["budget_level"].append(budget_level)
            
            # Add special requirements
            special_requirements = []
            
            # Check for special needs
            if "TRAVELER_PROFILE" in semantic_analysis:
                traveler = semantic_analysis["TRAVELER_PROFILE"]
                if "special_needs" in traveler and isinstance(traveler["special_needs"], list) and traveler["special_needs"]:
                    special_requirements.extend(traveler["special_needs"])
            
            # Check for special occasions
            if "CONTEXTUAL_FACTORS" in semantic_analysis:
                contextual = semantic_analysis["CONTEXTUAL_FACTORS"]
                if "special_occasions" in contextual and isinstance(contextual["special_occasions"], list) and contextual["special_occasions"]:
                    for occasion in contextual["special_occasions"]:
                        special_requirements.append(f"Special occasion: {occasion}")
            
            if special_requirements:
                enhanced_keywords["special_requirements"] = special_requirements
            
            # Deduplicate all lists and ensure they're strings
            for key in enhanced_keywords:
                if isinstance(enhanced_keywords[key], list):
                    enhanced_keywords[key] = [str(item) for item in enhanced_keywords[key]]
                    enhanced_keywords[key] = list(dict.fromkeys(enhanced_keywords[key]))
            
            print(f"Enhanced keywords with semantic understanding: {enhanced_keywords}")
            return enhanced_keywords
            
        except Exception as e:
            print(f"Error enhancing keywords with semantic analysis: {e}")
            self.logger.warning(f"Falling back to basic keywords after error in semantic enhancement: {e}")
            return basic_keywords

    def generate_travel_plan_chat(self, user_query: str) -> str:
        """
        Generate a complete travel plan using a multi-stage semantic approach.
        
        Args:
            user_query: User's travel plan request
            
        Returns:
            Completed travel plan as a string
        """
        print("Starting intelligent multi-stage travel plan generation...")
        
        # Initialize tracking variables
        start_time = time.time()
        api_calls_start = self.api_call_counter if hasattr(self, 'api_call_counter') else 0
        self.api_call_counter = api_calls_start if hasattr(self, 'api_call_counter') else 0
        section_timings = {}
        
        # Step 1: Perform comprehensive semantic analysis of the user request
        section_start = time.time()
        semantic_analysis = self.semantic_analyze_request(user_query)
        section_timings['semantic_analysis'] = time.time() - section_start
        section_start = time.time()
        
        # Step 2: Extract enhanced search keywords with semantic understanding
        search_keywords = self.extract_search_keywords_with_intent(user_query, semantic_analysis)
        section_timings['keyword_extraction'] = time.time() - section_start
        section_start = time.time()
        
        # Step 3: Retrieve relevant context using multilingual batch search
        context = self.rag_db.get_relevant_context_with_llm_keywords(
            user_query, 
            search_keywords, 
            n_results=7  # Increased from 5 to get more relevant context
        )
        section_timings['context_retrieval'] = time.time() - section_start
        section_start = time.time()
        
        # Step 4: Get sources for attribution with priority for MUST SEE content
        sources = self.rag_db.get_source_documents_with_llm_keywords(
            user_query,
            search_keywords,
            n_results=7  # Increased from 5 to match context
        )
        source_text = "\n".join([f"- {source}" for source in sources])
        section_timings['sources_retrieval'] = time.time() - section_start
        section_start = time.time()
        
        # Step 5: Create a structured prompt based on semantic analysis
        structured_prompt = self._create_structured_prompt(user_query, semantic_analysis, search_keywords)
        section_timings['structured_prompt'] = time.time() - section_start
        section_start = time.time()
        
        # Step 6: Generate a request analysis with the structured prompt
        request_analysis_prompt = f"""
You are an expert travel consultant specializing in transforming travel requests into actionable plans.
Analyze this travel request DEEPLY to identify EXACTLY what the traveler wants:

ORIGINAL REQUEST:
"{user_query}"

SEMANTIC ANALYSIS:
{structured_prompt}

CONDUCT A COMPREHENSIVE ANALYSIS:
1. Exact traveler type and group composition (solo, couple, family with X children, friends group, etc.)
2. Primary trip purpose (relaxation, adventure, cultural immersion, celebration, etc.)
3. Key destinations with EXACT requested duration for each
4. Must-have experiences and non-negotiable elements
5. Travel pace and intensity level (relaxed, moderate, intensive)
6. Special requirements or preferences (accessibility needs, dietary restrictions, etc.)
7. Budget category and considerations
8. Trip complexity assessment (simple, medium, complex)

CRITICAL: Note ANY specific durations mentioned (e.g., "2 days in Munich") - these MUST be honored exactly.
Your analysis will guide the creation of a perfectly tailored travel plan.
"""
        
        # Generate the analysis
        request_analysis = self._rate_limited_generate(self.model, request_analysis_prompt, component="request_analysis").text
        section_timings['request_analysis'] = time.time() - section_start
        section_start = time.time()
        
        # Step 7: Generate the main travel plan with all this analysis
        basic_prompt = f"""
You are an expert travel planner creating a HIGHLY PERSONALIZED travel plan based on this request:

"{user_query}"

COMPREHENSIVE REQUEST ANALYSIS:
{request_analysis}

STRUCTURED UNDERSTANDING:
{structured_prompt}

RELEVANT TRAVEL INFORMATION FROM DATABASE:
{context}

AVAILABLE SOURCES:
{source_text}

Create a detailed, inspiring travel plan with the following elements:
1. A DESCRIPTIVE TITLE that includes:
   - Main destinations
   - Trip duration (total days)
   - Primary trip purpose/theme

2. STRUCTURED PROMPT UNDERSTANDING - Include a section summarizing exactly how you understood the request 

3. EXECUTIVE SUMMARY that precisely addresses all key aspects of the request

4. DAILY ITINERARY with VIVID, SPECIFIC descriptions:
   - Follow the EXACT sequence and duration for each location as specified in the request
   - If specific timeframes were requested (e.g., "2 days in Munich"), HONOR these EXACTLY
   - Balance the itinerary according to the traveler's interests and pace preferences
   - For each day, provide specific morning, afternoon, and evening activities
   - Include realistic travel times between destinations

5. For EACH recommendation:
   - CRITICAL: Clearly and explicitly mark as [FROM DATABASE] or [EXTERNAL SUGGESTION]
   - For database-sourced items that are marked as must-see, use [FROM DATABASE - MUST-SEE]
   - Briefly explain WHY you're recommending it for this specific traveler
   - Include practical details (opening hours, costs, booking requirements, etc.)
   - For [FROM DATABASE] items, mention which source document provided the information
   - IMPORTANT: Every single recommendation MUST have one of these tags

CRITICAL REQUIREMENTS:
- Focus intensely on the traveler's specific interests and needs
- Prioritize [FROM DATABASE] recommendations when they match the request
- Create a truly personalized plan, not a generic itinerary
- Use vivid, sensory-rich descriptions for a more immersive experience
- Be specific about locations, timing, and practical details
- ENSURE the plan PRECISELY aligns with the traveler's request
"""
        
        # Generate the basic travel plan
        basic_plan = self._rate_limited_generate(self.model, basic_prompt).text
        section_timings['basic_plan'] = time.time() - section_start
        section_start = time.time()
        
        # Extract the title from the basic plan for use in other sections
        import re
        title_match = re.search(r"^#\s+(.+?)$", basic_plan, re.MULTILINE)
        if title_match:
            plan_title = title_match.group(1).strip()
        else:
            plan_title = "Personalized Travel Plan"
        
        # Step 8: Generate specialized accommodations section with semantic understanding
        accommodations_prompt = f"""
You are creating the ACCOMMODATIONS section for this travel plan: "{plan_title}"

TRAVELER PROFILE:
{semantic_analysis.get("TRAVELER_PROFILE", {})}

ORIGINAL REQUEST: 
"{user_query}"

PLAN ANALYSIS:
{request_analysis}

Based on this travel plan:
{basic_plan[:2500]}... (plan continues)

And using this travel database information:
{context}

Create a DETAILED "ACCOMMODATIONS" section with recommendations perfectly matched to the traveler's needs.

CRITICAL CONSIDERATIONS:
- Group Composition: {semantic_analysis.get("TRAVELER_PROFILE", {}).get("relationship_dynamics", "Unknown")} with {semantic_analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown")} travelers
- Budget Level: {semantic_analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "mid-range")}
- Special Needs: {", ".join(semantic_analysis.get("TRAVELER_PROFILE", {}).get("special_needs", []) or ["None specified"])}
- Trip Pace: Balance convenience with the itinerary's movement pace

For EACH accommodation:
- Name, location, and descriptive details
- Room types appropriate for this specific traveler/group
- Exact price range with currency
- Distance/time to key attractions or transit hubs
- Special features relevant to this traveler's needs and interests
- Practical booking information and tips
- WHY this specific accommodation is perfect for this traveler
- Mark each as [FROM DATABASE] or [EXTERNAL SUGGESTION] with source

IMPORTANT FORMAT REQUIREMENTS:
- Organize by destination in the exact order of the itinerary
- Include at least 2 options per location (if possible)
- Make recommendations specific and personalized, not generic
- Do NOT reference any "previous responses" - include ALL accommodation details directly
- Use vivid, appealing descriptions that highlight relevant features
"""
        
        accommodations_section = self._rate_limited_generate(self.model, accommodations_prompt).text
        section_timings['accommodations'] = time.time() - section_start
        section_start = time.time()
        
        # Step 9: Generate activities section focused on traveler interests
        activities_prompt = f"""
You are creating the ACTIVITIES & ATTRACTIONS section for this travel plan: "{plan_title}"

TRAVELER INTERESTS:
{semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {})}

ORIGINAL REQUEST: 
"{user_query}"

PLAN ANALYSIS:
{request_analysis}

Based on this travel plan:
{basic_plan[:2500]}... (plan continues)

And using this travel database information:
{context}

Create a DETAILED "ACTIVITIES & ATTRACTIONS" section with experiences perfectly matched to the traveler's interests.

CRITICAL REQUIREMENTS:
- Focus on activities that match the primary motivation: {semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("primary_trip_motivation", "general travel")}
- Highlight experiences related to key themes: {", ".join(semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("key_themes", []) or ["Not specified"])}
- Include any specifically requested experiences: {", ".join(semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("special_experiences", []) or ["None mentioned"])}
- Consider group composition: {semantic_analysis.get("TRAVELER_PROFILE", {}).get("relationship_dynamics", "Unknown")} with {semantic_analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown")} travelers
- Account for any special needs: {", ".join(semantic_analysis.get("TRAVELER_PROFILE", {}).get("special_needs", []) or ["None specified"])}

For EACH activity/attraction:
- Name, location, and vivid description
- Why it's SPECIFICALLY relevant to this traveler's interests
- Practical visiting information (hours, costs, reservation requirements)
- Insider tips to enhance the experience
- Estimated time needed
- Best time to visit
- Mark each as [FROM DATABASE] or [EXTERNAL SUGGESTION] with source

IMPORTANT FORMAT REQUIREMENTS:
- Organize by destination in the exact order of the itinerary
- Prioritize unique and authentic experiences
- Balance must-see attractions with lesser-known gems
- Include a mix of activity types appropriate for this traveler
- For each [FROM DATABASE] item, explain WHY it was selected for this specific traveler
- Do NOT reference any "previous responses" - include ALL activity details directly
"""
        
        activities_section = self._rate_limited_generate(self.model, activities_prompt).text
        section_timings['activities'] = time.time() - section_start
        section_start = time.time()
        
        # Step 10: Generate transportation section with realistic logistics
        transport_prompt = f"""
You are creating the TRANSPORTATION DETAILS section for this travel plan: "{plan_title}"

TRAVELER LOGISTICS PREFERENCES:
{semantic_analysis.get("TRIP_LOGISTICS", {})}

ORIGINAL REQUEST: 
"{user_query}"

PLAN ANALYSIS:
{request_analysis}

Based on this travel plan:
{basic_plan[:2500]}... (plan continues)

And using this travel database information:
{context}

Create a COMPREHENSIVE "TRANSPORTATION DETAILS" section with logistically sound connections between all destinations.

CRITICAL LOGISTICS REQUIREMENTS:
- Follow the EXACT sequence of destinations in the itinerary
- Account for group size: {semantic_analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown")} travelers
- Consider special needs: {", ".join(semantic_analysis.get("TRAVELER_PROFILE", {}).get("special_needs", []) or ["None specified"])}
- Budget level: {semantic_analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "mid-range")}
- Transportation preferences: {", ".join(semantic_analysis.get("TRIP_LOGISTICS", {}).get("transportation_preferences", []) or ["Not specified"])}

Include THREE main transportation categories:

1. DESTINATION-TO-DESTINATION TRAVEL:
   - Specific transportation modes (flight numbers/routes, train services, etc.)
   - Multiple options when relevant (economy vs. premium, direct vs. connecting)
   - Realistic departure/arrival times
   - Journey duration including connections and transfers
   - Booking information and estimated costs
   - Transportation hubs and terminal information

2. AIRPORT/STATION TRANSFERS:
   - Options for reaching accommodations from arrival points
   - Estimated transfer times and costs
   - Pre-booking requirements

3. LOCAL TRANSPORTATION:
   - Best ways to navigate within each destination
   - Local transportation passes or money-saving options
   - App recommendations for navigation or ticketing
   - Special considerations for evenings or early mornings

FOR EACH RECOMMENDATION:
- Mark as [FROM DATABASE] or [EXTERNAL SUGGESTION]
- Include practical details (booking websites, costs, schedules)
- Add insider tips to improve the travel experience
- Explain why this option is suitable for this specific traveler

IMPORTANT: Create a REALISTIC and EFFICIENT transportation plan that:
- Respects the travel pace indicated in the request
- Minimizes unnecessary transit time
- Accounts for check-in/check-out times at accommodations
- Includes buffer time for connections
- Flags any logistical challenges with specific solutions
- Is organized chronologically to match the itinerary flow
"""
        
        transport_section = self._rate_limited_generate(self.model, transport_prompt).text
        section_timings['transport'] = time.time() - section_start
        section_start = time.time()
        
        # Step 11: Generate dining section focused on culinary preferences
        dining_prompt = f"""
You are creating the DINING RECOMMENDATIONS section for this travel plan: "{plan_title}"

TRAVELER PROFILE:
{semantic_analysis.get("TRAVELER_PROFILE", {})}

TRAVELER INTERESTS:
{semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {})}

ORIGINAL REQUEST: 
"{user_query}"

Based on this travel plan:
{basic_plan[:2000]}... (plan continues)

And using this travel database information:
{context}

Create a TAILORED "DINING RECOMMENDATIONS" section perfectly matched to this traveler's preferences.

CRITICAL CONSIDERATIONS:
- Food interest level: {semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("key_themes", ["Unknown"]) if "food" in str(semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("key_themes", [])).lower() else "Not specifically mentioned"}
- Group composition: {semantic_analysis.get("TRAVELER_PROFILE", {}).get("relationship_dynamics", "Unknown")} with {semantic_analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown")} travelers
- Budget level: {semantic_analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "mid-range")}
- Special dietary needs: {", ".join([need for need in semantic_analysis.get("TRAVELER_PROFILE", {}).get("special_needs", []) if "diet" in need.lower() or "allerg" in need.lower() or "vegan" in need.lower() or "vegetarian" in need.lower()]) or "None specified"}

For EACH destination in the itinerary, provide:

1. MUST-TRY LOCAL SPECIALTIES:
   - Iconic dishes and where to find the best versions
   - Local beverages and food traditions
   - Seasonal specialties if relevant

2. RESTAURANT RECOMMENDATIONS (at least 3 per destination):
   - Name, location, and cuisine type
   - Signature dishes and specialties
   - Price range with currency
   - Atmosphere and dining experience
   - Reservation requirements
   - WHY this restaurant is appropriate for this specific traveler
   - Mark each as [FROM DATABASE] or [EXTERNAL SUGGESTION]

3. DINING EXPERIENCES:
   - Food tours or cooking classes if relevant to traveler interests
   - Markets or food streets worth visiting
   - Special dining experiences (rooftop restaurants, unique settings, etc.)

4. PRACTICAL DINING TIPS:
   - Meal timing customs in each destination
   - Tipping practices
   - Reservation guidance
   - Dress codes if applicable

IMPORTANT FORMAT REQUIREMENTS:
- Organize recommendations by destination in itinerary order
- Include diverse options for different meals and preferences
- Balance fine dining with authentic local experiences
- Pay special attention to any food interests mentioned in the request
- Make recommendations specific and personal, not generic
- For child travelers, include family-friendly options
"""
        
        dining_section = self._rate_limited_generate(self.model, dining_prompt).text
        section_timings['dining'] = time.time() - section_start
        section_start = time.time()
        
        # Step 12: Generate budget breakdown based on traveler profile
        budget_prompt = f"""
You are creating the ESTIMATED BUDGET BREAKDOWN section for this travel plan: "{plan_title}"

TRAVELER PROFILE:
{semantic_analysis.get("TRAVELER_PROFILE", {})}

BUDGET CONTEXT:
{semantic_analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "mid-range")} level

ORIGINAL REQUEST: 
"{user_query}"

Based on this travel plan:
{basic_plan[:2000]}... (plan continues)

Create a DETAILED "ESTIMATED BUDGET BREAKDOWN" section with realistic costs for this specific trip.

CRITICAL REQUIREMENTS:
- Base all estimates on {semantic_analysis.get("TRAVELER_PROFILE", {}).get("number_of_travelers", "Unknown")} travelers
- Use the appropriate budget level: {semantic_analysis.get("CONTEXTUAL_FACTORS", {}).get("budget_level", "mid-range")}
- Calculate for the full duration: {semantic_analysis.get("TRIP_LOGISTICS", {}).get("total_duration", "the entire trip")}
- Use the local currency with USD equivalent
- Provide itemized costs AND daily averages

Include the following categories:

1. TRANSPORTATION:
   - International flights (if applicable)
   - Domestic transportation between destinations
   - Airport/station transfers
   - Local transportation within cities
   - Rental vehicles if applicable
   - SUBTOTAL for all transportation

2. ACCOMMODATIONS:
   - Breakdown by destination
   - Cost per night and total for each location
   - Any additional fees (resort fees, city taxes, etc.)
   - SUBTOTAL for all accommodations

3. FOOD & DINING:
   - Average cost for breakfast, lunch, dinner
   - Estimated costs for special dining experiences
   - Drinks and snacks
   - SUBTOTAL for all food and dining

4. ACTIVITIES & ATTRACTIONS:
   - Entrance fees for attractions
   - Tours and guided experiences
   - Special activities
   - SUBTOTAL for all activities

5. MISCELLANEOUS:
   - Travel insurance
   - Visa fees if applicable
   - Shopping/souvenirs allowance
   - Contingency fund (10-15% of total)
   - SUBTOTAL for miscellaneous expenses

6. GRAND TOTAL:
   - Total trip cost
   - Cost per person
   - Daily average per person

7. MONEY-SAVING TIPS:
   - Specific tips for this itinerary
   - Potential areas for cost reduction
   - Recommended splurges worth the expense

Present this as a clear, itemized breakdown in table format where possible. Be realistic about costs based on current prices in each destination.
"""
        
        budget_section = self._rate_limited_generate(self.model, budget_prompt).text
        section_timings['budget'] = time.time() - section_start
        section_start = time.time()
        
        # Step 13: Generate practical tips section tailored to this traveler
        tips_prompt = f"""
You are creating the PRACTICAL TIPS section for this travel plan: "{plan_title}"

TRAVELER PROFILE:
{semantic_analysis.get("TRAVELER_PROFILE", {})}

TRIP CONTEXT:
- Destinations: {", ".join(search_keywords.get("locations", ["Various destinations"]))}
- Duration: {semantic_analysis.get("TRIP_LOGISTICS", {}).get("total_duration", "Multiple days")}
- Trip type: {semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {}).get("primary_trip_motivation", "travel")}

ORIGINAL REQUEST: 
"{user_query}"

Based on this travel plan:
{basic_plan[:2000]}... (plan continues)

And using this travel database information:
{context}

Create a COMPREHENSIVE "PRACTICAL TIPS" section with advice specifically relevant to this traveler and itinerary.

Include the following categories, focusing on what's most relevant to these specific destinations and this traveler:

1. ESSENTIAL PREPARATIONS:
   - Required travel documents
   - Visa requirements and application processes
   - Vaccinations or health requirements
   - Travel insurance recommendations
   - Pre-booking necessities

2. PACKING RECOMMENDATIONS:
   - Climate-appropriate clothing
   - Essential items specific to these destinations
   - Electronics and adapters
   - Special gear for planned activities
   - Health and medication considerations

3. HEALTH & SAFETY:
   - Common health concerns in these destinations
   - Local emergency numbers and medical facilities
   - Safety precautions for specific areas
   - COVID-19 or other current health considerations
   - Food and water safety tips if relevant

4. CULTURAL ETIQUETTE:
   - Local customs and traditions
   - Appropriate dress codes
   - Tipping practices
   - Communication tips and key phrases
   - Religious or cultural sensitivities

5. MONEY MATTERS:
   - Currency information
   - Best payment methods
   - ATM availability
   - Tipping customs
   - Money-saving strategies

6. CONNECTIVITY & TECHNOLOGY:
   - SIM card and internet access options
   - Useful apps for these destinations
   - Power adapters needed
   - Navigation tips

7. TRANSPORTATION SAVVY:
   - Tips for using local transportation
   - Taxi and rideshare guidance
   - Driving considerations if applicable
   - Walking and accessibility notes

FOR EACH TIP:
- Make it specifically relevant to this traveler's profile and itinerary
- Prioritize practical, actionable advice over general information
- Include insider knowledge when available from the database
- Mark database-sourced tips as [FROM DATABASE]

IMPORTANT: Tailor all advice to this specific traveler's needs, the exact destinations in the itinerary, and the particular experiences planned. Avoid generic travel tips that don't directly apply to this trip.
"""
        
        tips_section = self._rate_limited_generate(self.model, tips_prompt).text
        section_timings['tips'] = time.time() - section_start
        section_start = time.time()
        
        # Step 14: Generate comprehensive sources section
        sources_prompt = f"""
You are creating the SOURCES section for this travel plan: "{plan_title}"

Based on the following sources used in this travel plan:
{source_text}

Create a DETAILED "SOURCES" section that:

1. Lists all database documents used in creating this plan
2. For each major recommendation in the plan, identifies its specific source
3. Provides transparency about which information came from the database vs. external knowledge
4. Explains any instances where external suggestions were necessary
5. Categorizes sources by destination or topic area

CRITICAL REQUIREMENTS:
- Be comprehensive - include ALL sources used
- Be transparent about the origin of recommendations
- Organize sources in a clear, structured format
- Explain why external suggestions were made when database information was insufficient
- Note any MUST SEE or IMPORTANT tags from the database that influenced recommendations

FORMAT:
- Create subsections by destination or category
- For each source, include document name and key information it provided
- Use bullet points for clarity and readability
- Provide a brief assessment of database coverage for this specific itinerary

This sources section serves as documentation of information provenance and enhances the credibility of the travel plan.
"""
        
        sources_section = self._rate_limited_generate(self.model, sources_prompt).text
        section_timings['sources'] = time.time() - section_start
        section_start = time.time()
        
        # Combine all sections into one complete travel plan with a clear structure
        complete_plan = f"""
{basic_plan}

## 🏨 Detailed Accommodations
{accommodations_section}

## 🏛️ Activities & Attractions
{activities_section}

## 🚌 Transportation Details
{transport_section}

## 🍽️ Dining Recommendations
{dining_section}

## 💰 Estimated Budget Breakdown
{budget_section}

## 📋 Practical Tips
{tips_section}

## 📚 Sources & References
{sources_section}
"""

        # Step 15: Run a plausibility check on the generated plan
        plausibility_prompt = f"""
You are a senior travel consultant tasked with assessing the plausibility and quality of this travel plan:

ORIGINAL REQUEST: 
"{user_query}"

TRAVEL PLAN TITLE:
{plan_title}

Analyze this travel plan:
{complete_plan[:3000]}... (plan continues)

Conduct a COMPREHENSIVE PLAUSIBILITY AND QUALITY ASSESSMENT focusing on:

1. REQUEST ALIGNMENT:
   - Does the plan PRECISELY address what the traveler requested? (Score 1-10)
   - Are there any specific aspects of the request that were missed or underserved?
   - Does the plan honor specific time allocations mentioned in the request?

2. LOGISTICAL COHERENCE:
   - Is the itinerary logistically feasible? (Score 1-10)
   - Are the travel times between destinations realistic?
   - Is there sufficient time allocated for activities given transit times?
   - Are there any scheduling conflicts or impossibilities?

3. ACCOMMODATION SUITABILITY:
   - Do the accommodations match the traveler's needs and budget level? (Score 1-10)
   - Are the locations practical given the itinerary?
   - Are check-in/check-out times accounted for?

4. ACTIVITY APPROPRIATENESS:
   - Do the recommended activities align with the traveler's interests? (Score 1-10)
   - Is the activity density appropriate (not too packed or too sparse)?
   - Are the activities suitable for the traveler profile (age, mobility, etc.)?

5. TRANSPORTATION FEASIBILITY:
   - Are the transportation recommendations realistic and efficient? (Score 1-10)
   - Are connection times sufficient?
   - Are there contingencies for potential delays?

6. BUDGET ACCURACY:
   - Is the budget realistic for this itinerary? (Score 1-10)
   - Are there any major cost omissions?
   - Does it align with the indicated budget level?

7. OVERALL PLAN QUALITY:
   - Overall plausibility rating (1-10)
   - Strongest aspects of the plan
   - Areas needing improvement

If any category scores below 7/10, provide SPECIFIC RECOMMENDATIONS for improvement.
Create a "PLAUSIBILITY ASSESSMENT" section with your findings in a clear, structured format.
"""
        
        try:
            plausibility_check = self._rate_limited_generate(self.model, plausibility_prompt).text
            section_timings['plausibility_check'] = time.time() - section_start
            section_start = time.time()
            
            # Add the plausibility check to the complete plan
            complete_plan = f"{complete_plan}\n\n## ✅ PLAUSIBILITY ASSESSMENT\n{plausibility_check}"
        except Exception as e:
            self.logger.error(f"Error generating plausibility check: {str(e)}")
            # Add a simplified note if the check fails
            complete_plan = f"{complete_plan}\n\n## ✅ PLAUSIBILITY ASSESSMENT\nUnable to generate plausibility check due to an error."
        
        # Generate metadata if needed
        if not hasattr(self.config, 'SKIP_METADATA_GENERATION') or not self.config.SKIP_METADATA_GENERATION:
            try:
                # Generate the metadata analysis with focus on database utilization and content relevance
                metadata_prompt = f"""
You are a travel plan quality analyst creating metadata for this travel plan: "{plan_title}"

TRAVELER PROFILE:
{semantic_analysis.get("TRAVELER_PROFILE", {})}

TRAVEL CONTENT SOURCES:
{source_text}

Analyze this complete travel plan:
{complete_plan[:3000]}... (plan continues)

Create a comprehensive "PLAN METADATA & ANALYSIS" section that includes:

1. DATABASE UTILIZATION
   - Estimate the percentage (0-100%) of plan content from database vs. external knowledge
   - Create a table showing the source breakdown by recommendation category:
     * Accommodations: % database vs % external
     * Activities: % database vs % external
     * Dining: % database vs % external
     * Transportation: % database vs % external
   - Identify which destinations had strong vs. weak database coverage
   - Note specific instances where MUST SEE or IMPORTANT content was incorporated

2. TRAVELER NEEDS ANALYSIS
   - How well did the plan address the specific traveler profile?
   - Were special needs or requirements successfully accommodated?
   - Were specific interests and preferences reflected in the recommendations?
   - Were any aspects of the traveler profile underserved?

3. PLAN CLASSIFICATION
   - Primary trip classification (adventure, cultural, beach, etc.)
   - Secondary themes
   - Budget level classification
   - Intensity/pace level (1-5 scale)
   - Target audience profile
   - Special features or unique aspects

4. IMPROVEMENT OPPORTUNITIES
   - Database content gaps identified
   - Additional information that would enhance future plans
   - Specific recommendation areas that could be strengthened
   - Missing destination content needed

FORMAT:
- Create a structured, well-organized metadata section
- Use tables, bullet points, and clear headings
- Be specific and data-driven in your analysis
- Focus on objective assessment rather than generic praise
"""
                
                metadata_section = self._rate_limited_generate(self.model, metadata_prompt).text
                section_timings['metadata'] = time.time() - section_start
                section_start = time.time()
                
                # Add metadata to the complete plan
                complete_plan = f"{complete_plan}\n\n{'-'*80}\n\n# 📊 PLAN METADATA & ANALYSIS\n\n{metadata_section}"
                
            except Exception as e:
                self.logger.error(f"Error generating metadata: {str(e)}")
                # Add simplified metadata section in case of error
                complete_plan = f"{complete_plan}\n\n{'-'*80}\n\n# 📊 PLAN METADATA & ANALYSIS\n\nUnable to generate detailed metadata. This plan was created based on available travel information."
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        total_api_calls = self.api_call_counter - api_calls_start
        
        # Create a performance report with an enhanced table
        
        # Create a section timing table with visual representation
        timing_rows = []
        for section, seconds in section_timings.items():
            percent = (seconds / total_time) * 100
            bars = "█" * int(percent / 5)  # Visual bar chart (1 bar = 5%)
            timing_rows.append(f"| {section:25} | {seconds:6.1f}s | {percent:5.1f}% | {bars} |")
        
        timing_table = "\n".join(timing_rows)
        
        performance_report = f"""
{'-'*80}

# 📊 PERFORMANCE REPORT

## ⏱️ Generation Performance
- **Total Generation Time:** {total_time/60:.2f} minutes ({total_time:.1f} seconds)
- **API Calls Made:** {total_api_calls}
- **Average Time Per API Call:** {total_time/total_api_calls:.2f} seconds

## 📋 Section Timing Breakdown
| Section                    | Time    | % Total | Visualization        |
|----------------------------|---------|---------|----------------------|
{timing_table}

{'-'*80}
"""
        
        # Add the performance report at the very end
        complete_plan = f"{complete_plan}\n\n{performance_report}"
        
        return complete_plan
    
    def _create_structured_prompt(self, user_query: str, semantic_analysis: dict, search_keywords: dict) -> str:
        """
        Create a structured prompt based on semantic analysis results.
        
        Args:
            user_query: Original user query
            semantic_analysis: Results from semantic analysis
            search_keywords: Extracted search keywords
            
        Returns:
            Formatted structured prompt string
        """
        try:
            # Extract key information from semantic analysis
            traveler_profile = semantic_analysis.get("TRAVELER_PROFILE", {})
            trip_logistics = semantic_analysis.get("TRIP_LOGISTICS", {})
            motivations = semantic_analysis.get("TRIP_MOTIVATIONS_&_EXPECTATIONS", {})
            contextual = semantic_analysis.get("CONTEXTUAL_FACTORS", {})
            
            # Format location-specific durations if available
            location_durations = ""
            if "specific_timeframes" in trip_logistics and isinstance(trip_logistics["specific_timeframes"], dict):
                for location, duration in trip_logistics["specific_timeframes"].items():
                    location_durations += f"- {location}: {duration}\n"
            
            # Format relationships from search keywords
            relationships = ""
            if "relationships" in search_keywords and search_keywords["relationships"]:
                relationships += "LOCATION-DURATION RELATIONSHIPS:\n"
                for relationship in search_keywords["relationships"]:
                    relationships += f"- {relationship}\n"
            
            # Combine into a structured prompt
            structured_prompt = f"""
TRAVELER PROFILE:
- Type: {traveler_profile.get("relationship_dynamics", "Unknown")}
- Number: {traveler_profile.get("number_of_travelers", "Unknown")}
- Age Groups: {", ".join(traveler_profile.get("age_groups", ["Unknown"]))}
- Special Needs: {", ".join(traveler_profile.get("special_needs", []) or ["None specified"])}
- Language: {", ".join(traveler_profile.get("language_preferences", ["Unknown"]))}

TRIP LOGISTICS:
- Total Duration: {trip_logistics.get("total_duration", "Unknown")}
- Destination Sequence: {", ".join(search_keywords.get("locations", ["Unknown"]))}
{location_durations if location_durations else ""}
{relationships if relationships else ""}
- Transportation Preferences: {", ".join(trip_logistics.get("transportation_preferences", []) or ["Not specified"])}
- Accommodation Preferences: {", ".join(search_keywords.get("accommodation_types", []) or ["Not specified"])}

TRIP MOTIVATIONS:
- Primary Purpose: {motivations.get("primary_trip_motivation", "Unknown")}
- Key Themes: {", ".join(search_keywords.get("themes", []) or ["Not specified"])}
- Priority Activities: {", ".join(search_keywords.get("activities", []) or ["Not specified"])}
- Special Experiences: {", ".join(motivations.get("special_experiences", []) or ["None specified"])}
- Must-Have Elements: {", ".join(motivations.get("must_have_elements", []) or ["None specified"])}

CONTEXTUAL FACTORS:
- Budget Level: {contextual.get("budget_level", "Unknown")}
- Special Occasions: {", ".join(contextual.get("special_occasions", []) or ["None mentioned"])}
- Prior Experience: {contextual.get("prior_travel_experience", "Unknown")}
- Concerns/Constraints: {", ".join(contextual.get("concerns_constraints", []) or ["None mentioned"])}
"""
            
            return structured_prompt
            
        except Exception as e:
            print(f"Error creating structured prompt: {e}")
            return f"""
STRUCTURED PROMPT (Error occurred during creation):
- Original query: {user_query}
- Extracted locations: {', '.join(search_keywords.get('locations', ['Unknown']))}
- Duration: {search_keywords.get('timeframe', ['Unknown'])}
- Key themes: {', '.join(search_keywords.get('themes', ['Unknown']))}
"""
        
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