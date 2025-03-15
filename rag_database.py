"""
RAG database module for the Travel Plan Agent.
This module contains functionality for managing the vector database using ChromaDB.
"""

import re
import time
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class RAGDatabase:
    """Class for managing the RAG vector database with ChromaDB."""
    
    def __init__(self, config, force_recreate=True):
        """
        Initialize the RAG database.
        
        Args:
            config: Application configuration object
            force_recreate: Whether to force recreate the collection
        """
        self.config = config
        
        # Use an in-memory client instead of persistent due to permission issues
        # This will work for the demo but data will be lost when the application is closed
        print("Using in-memory ChromaDB client for demo purposes")
        self.client = chromadb.Client()
        
        # Set up default embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        
        # First try to delete any existing collection
        if force_recreate:
            try:
                self.client.delete_collection(name="travel_documents")
                print("Deleted existing travel_documents collection for clean start")
            except Exception as e:
                print(f"No existing collection to delete or error: {e}")
        
        try:
            # Try to get the existing collection
            self.collection = self.client.get_collection(
                name="travel_documents",
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: travel_documents")
        except Exception as e:
            print(f"Creating new collection: {e}")
            # Create a new collection
            self.collection = self.client.create_collection(
                name="travel_documents",
                embedding_function=self.embedding_function,
                metadata={"description": "Travel documents for personalized travel planning"}
            )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            # Get the count of documents in the collection
            count = self.collection.count()
            
            # Get other collection information
            info = {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata
            }
            
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"name": "unknown", "count": 0, "metadata": {}}
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents in the vector database.
        
        Args:
            documents: List of document chunks with metadata
        """
        if not documents:
            print("No documents to index.")
            return
        
        # Prepare documents for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # Ensure we have valid text content
            if not doc.get('text'):
                print(f"Skipping document {i} - no text content")
                continue
                
            # Create a unique ID
            doc_id = f"doc_{i}_{doc['metadata']['filename']}_{doc['metadata']['chunk_id']}"
            
            # Diagnostic info about the document being indexed
            print(f"Indexing document: {doc_id}")
            print(f"Text length: {len(doc['text'])} characters")
            print(f"First 100 chars: {doc['text'][:100]}...")
            
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(doc['metadata'])
        
        # Add documents to the collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            try:
                # Record timestamp before adding batch
                start_time = time.time()
                
                # Add the batch to the collection
                try:
                    self.collection.add(
                        ids=ids[i:batch_end],
                        documents=texts[i:batch_end],
                        metadatas=metadatas[i:batch_end]
                    )
                    
                    # Calculate time taken
                    elapsed = time.time() - start_time
                    print(f"Batch {i//batch_size + 1} indexed in {elapsed:.2f} seconds ({batch_end - i} documents)")
                except Exception as first_e:
                    print(f"Error indexing batch {i//batch_size + 1}: {first_e}")
                    
                    # If we hit a read-only database error, we need to recreate the collection
                    if "readonly database" in str(first_e).lower():
                        print("Hit readonly database error, attempting to recreate collection...")
                        try:
                            # Try to recreate collection
                            self.client.delete_collection("travel_documents")
                            self.collection = self.client.create_collection(
                                name="travel_documents",
                                embedding_function=self.embedding_function
                            )
                            print("Successfully recreated collection")
                            
                            # Try the add again
                            self.collection.add(
                                ids=ids[i:batch_end],
                                documents=texts[i:batch_end],
                                metadatas=metadatas[i:batch_end]
                            )
                            elapsed = time.time() - start_time
                            print(f"Batch {i//batch_size + 1} indexed in {elapsed:.2f} seconds ({batch_end - i} documents) after recreation")
                        except Exception as inner_e:
                            print(f"Failed to recover from readonly error: {inner_e}")
                
            except Exception as e:
                print(f"Outer error in batch {i//batch_size + 1}: {e}")
                # Continue with next batch even if this one fails
        
        # Verify documents were added
        count = self.collection.count()
        print(f"Collection now contains {count} document chunks")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        # First check if the query contains location names to prioritize
        import re
        locations = re.findall(r'\b([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)\b', query)
        common_non_locations = {'I', 'My', 'Me', 'Mine', 'The', 'A', 'An', 'And', 'Or', 'But', 'For', 'With', 'To', 'From'}
        locations = [loc for loc in locations if loc not in common_non_locations]
        
        # Determine if we need to modify the query to emphasize specific locations
        emphasized_query = query
        if locations:
            # Create an emphasized query that doubles the location mentions for better matching
            for location in locations:
                if location.lower() not in emphasized_query.lower():
                    emphasized_query = f"{emphasized_query} {location} {location}"
        
        # Perform the search with the potentially modified query
        print(f"Searching with query: '{emphasized_query}'")
        results = self.collection.query(
            query_texts=[emphasized_query],
            n_results=n_results
        )
        
        # Process and format the results
        formatted_results = []
        must_see_results = []  # Special container for must-see items
        important_results = [] # Medium priority items
        
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                score = results['distances'][0][i] if results['distances'] else None
                
                formatted_result = {
                    'text': doc,
                    'metadata': metadata,
                    'score': score,
                }
                
                # Check if document contains location names mentioned in the query
                location_match = False
                for location in locations:
                    if location.lower() in doc.lower():
                        location_match = True
                        break
                
                # Apply tiered prioritization logic
                if "has_must_see" in metadata and metadata["has_must_see"] == "yes":
                    # Top priority: MUST SEE content
                    must_see_results.append(formatted_result)
                    if "location_name" in metadata and metadata["location_name"]:
                        print(f"⭐⭐ Found MUST SEE information for: {metadata['location_name']}")
                    else:
                        print(f"⭐⭐ Found MUST SEE information in document")
                
                elif location_match:
                    # Medium priority: Content matching specific locations
                    important_results.append(formatted_result)
                    print(f"⭐ Found content matching location: {', '.join(locations)}")
                
                else:
                    # Regular priority content
                    formatted_results.append(formatted_result)
        
        # Modify n_results if we need to include more must-see content
        # This ensures that we don't lose must-see content due to the n_results limit
        if len(must_see_results) > 0:
            # Increase n_results to ensure at least 2 regular results along with all must-see content
            n_results = max(n_results, len(must_see_results) + 2)
        
        # Prioritize results in three tiers: must-see first, important second, regular last
        # But cap total results at the adjusted n_results
        combined_results = must_see_results + important_results + formatted_results
        return combined_results[:n_results]
    
    def extract_search_keywords(self, query: str) -> dict:
        """
        Extract important search keywords from a user query in multiple languages.
        
        Args:
            query: The user's search query
            
        Returns:
            Dictionary with keywords in different languages
        """
        # Translation dictionary for common travel terms
        translations = {
            # Locations are kept as-is across languages
            
            # Travel themes - German to English
            'strand': 'beach',
            'berg': 'mountain', 'berge': 'mountains', 'gebirge': 'mountains',
            'geschichte': 'history', 'historisch': 'historical',
            'kultur': 'culture', 'kulturell': 'cultural',
            'essen': 'food', 'kulinarisch': 'culinary', 'gastronomie': 'gastronomy',
            'abenteuer': 'adventure',
            'entspannung': 'relaxation', 'erholung': 'relaxation',
            'luxus': 'luxury',
            'budget': 'budget', 'günstig': 'budget',
            'wandern': 'hiking', 'trekking': 'trekking',
            'natur': 'nature',
            'stadt': 'city', 'städte': 'cities',
            'ländlich': 'rural',
            'tempel': 'temple',
            'museum': 'museum', 'museen': 'museums',
            'kunst': 'art',
            'festival': 'festival', 'fest': 'festival',
            'einkaufen': 'shopping',
            'reise': 'travel', 'urlaub': 'vacation',
            'hotels': 'hotels', 'hotel': 'hotel',
            'resort': 'resort', 'resorts': 'resorts',
            'flug': 'flight', 'flüge': 'flights',
            'strand': 'beach', 'strände': 'beaches',
            'insel': 'island', 'inseln': 'islands',
            'kreuzfahrt': 'cruise', 'kreuzfahrten': 'cruises'
        }
        
        # Define stopwords (words to ignore) in both languages
        german_stopwords = ['ich', 'mich', 'mir', 'mein', 'meine', 'meinen', 'meiner', 'meinem', 
                           'du', 'dich', 'dir', 'dein', 'deine', 'deinen', 'deiner', 'deinem',
                           'er', 'sie', 'es', 'ihn', 'ihm', 'ihr', 'sein', 'seine', 'seiner', 'seinem',
                           'wir', 'uns', 'unser', 'unsere', 'unseren', 'unserer', 'unserem',
                           'ihr', 'euch', 'euer', 'eure', 'euren', 'eurer', 'eurem',
                           'der', 'die', 'das', 'den', 'dem', 'des',
                           'ein', 'eine', 'einen', 'einem', 'einer', 'eines',
                           'und', 'oder', 'aber', 'denn', 'wenn', 'weil', 'als', 'wie',
                           'in', 'auf', 'unter', 'über', 'neben', 'zwischen', 'mit', 'ohne',
                           'zu', 'von', 'bei', 'nach', 'aus', 'vor', 'während', 'durch',
                           'für', 'gegen', 'um', 'an', 'am', 'im', 'zum', 'zur',
                           'ist', 'sind', 'war', 'waren', 'wird', 'werden', 'wurde', 'wurden',
                           'hat', 'haben', 'hatte', 'hatten', 'kann', 'können', 'möchte', 'möchten', 'will', 'wollen']
        
        english_stopwords = ['i', 'me', 'my', 'mine', 'myself',
                           'you', 'your', 'yours', 'yourself',
                           'he', 'him', 'his', 'himself',
                           'she', 'her', 'hers', 'herself',
                           'it', 'its', 'itself',
                           'we', 'us', 'our', 'ours', 'ourselves',
                           'they', 'them', 'their', 'theirs', 'themselves',
                           'the', 'a', 'an',
                           'and', 'or', 'but', 'because', 'if', 'when', 'as', 'like',
                           'in', 'on', 'under', 'over', 'beside', 'between', 'with', 'without',
                           'to', 'from', 'by', 'after', 'before', 'during', 'through',
                           'for', 'against', 'about', 'at',
                           'is', 'are', 'was', 'were', 'will', 'be', 'been',
                           'has', 'have', 'had', 'can', 'could', 'may', 'might', 'must', 'should',
                           'would', 'want', 'wanted', 'wants', 'go', 'going', 'goes', 'went']
        
        # Extract country and city names which are capitalized
        location_pattern = r'\b([A-Z][a-z]+)\b'
        locations = re.findall(location_pattern, query)
        
        # Filter out common capitalized words that aren't locations
        common_non_locations = ['I', 'My', 'Me', 'Mine', 'The', 'A', 'An', 'And', 'Or', 'But', 'For']
        locations = [loc for loc in locations if loc not in common_non_locations]
        
        # If "vietnam" is in query (case insensitive), add it to locations
        if 'vietnam' in query.lower() and 'Vietnam' not in locations:
            locations.append('Vietnam')
        
        # Basic English travel terms
        english_terms = ['beach', 'beaches', 'mountain', 'mountains', 'history', 'historical', 
                        'culture', 'cultural', 'food', 'culinary', 'adventure', 
                        'relaxation', 'luxury', 'budget', 'hiking', 'trekking', 
                        'nature', 'city', 'cities', 'rural', 'cruise', 'cruises',
                        'temple', 'museum', 'museums', 'art', 'festival', 'shopping',
                        'hotel', 'hotels', 'resort', 'resorts', 'villa', 'hostel',
                        'beach', 'beaches', 'island', 'islands', 'coast', 'coastal',
                        'cuisine', 'gastronomy', 'dining', 'restaurant', 'restaurants']
        
        # Basic German travel terms 
        german_terms = ['strand', 'strände', 'berg', 'berge', 'gebirge', 'geschichte', 'historisch',
                       'kultur', 'kulturell', 'essen', 'kulinarisch', 'gastronomie',
                       'abenteuer', 'entspannung', 'erholung', 'luxus', 'budget', 'günstig',
                       'wandern', 'trekking', 'natur', 'stadt', 'städte', 'ländlich',
                       'tempel', 'museum', 'museen', 'kunst', 'festival', 'fest', 'einkaufen',
                       'hotel', 'hotels', 'resort', 'resorts', 'kreuzfahrt', 'kreuzfahrten',
                       'insel', 'inseln', 'küste', 'küsten', 'strand', 'strände']
        
        # Find English terms (excluding stopwords)
        english_found = []
        for term in english_terms:
            if re.search(r'\b' + term + r'\b', query.lower()):
                if term not in english_stopwords:  # Only add if not a stopword
                    english_found.append(term)
        
        # Find German terms and translate them (excluding stopwords)
        german_found = []
        translated_german = []
        for term in german_terms:
            if re.search(r'\b' + term + r'\b', query.lower()):
                if term not in german_stopwords:  # Only add if not a stopword
                    german_found.append(term)
                    if term in translations:
                        translated_german.append(translations[term])
        
        # Combine all keywords
        keywords = {
            'locations': locations,
            'english_terms': english_found,
            'german_terms': german_found,
            'translated_german': translated_german
        }
        
        # If no specific keywords were found, extract general keywords
        if not (locations or english_found or german_found):
            # Get all words with 4+ characters as potential keywords, excluding stopwords
            words = query.lower().split()
            general_keywords = []
            
            for word in words:
                # Only consider words 4+ characters long
                if len(word) >= 4:
                    # Skip if it's in stopwords
                    if word not in german_stopwords and word not in english_stopwords:
                        general_keywords.append(word)
            
            if general_keywords:
                keywords['general'] = general_keywords
            else:
                # If still no keywords (e.g., query is all stopwords), use longest word as fallback
                longest_word = max(words, key=len) if words else ""
                if longest_word:
                    keywords['general'] = [longest_word]
        
        # Ensure we always have at least one search keyword
        if not any(keywords.values()):
            # Use the original query as a last resort
            keywords['general'] = [query.lower()]
        
        return keywords
    
    def enhanced_search(self, query: str, n_results: int = 5) -> list:
        """
        Perform an enhanced search by extracting keywords and doing multiple searches.
        
        Args:
            query: The search query
            n_results: Number of results per search
            
        Returns:
            Combined search results
        """
        print(f"Original search query: '{query}'")
        
        # First try with the original query
        original_results = self.search(query, n_results)
        
        # Extract keywords in multiple languages
        keyword_dict = self.extract_search_keywords(query)
        print(f"Extracted keywords: {keyword_dict}")
        
        # Look for food-related terms if not already found
        if not any(term in keyword_dict.get('english_terms', []) for term in ['food', 'culinary', 'cuisine', 'dining']):
            food_terms = ['food', 'eat', 'eating', 'cuisine', 'restaurant', 'meal', 'dining', 'gastronomy', 'culinary']
            for term in food_terms:
                if term in query.lower():
                    if 'english_terms' not in keyword_dict:
                        keyword_dict['english_terms'] = []
                    if 'food' not in keyword_dict['english_terms']:
                        keyword_dict['english_terms'].append('food')
                    break
        
        # Look for beach-related terms if not already found
        if not any(term in keyword_dict.get('english_terms', []) for term in ['beach', 'beaches', 'coastal']):
            beach_terms = ['beach', 'beaches', 'coastal', 'coast', 'shore', 'seaside', 'ocean', 'sea']
            for term in beach_terms:
                if term in query.lower():
                    if 'english_terms' not in keyword_dict:
                        keyword_dict['english_terms'] = []
                    if 'beach' not in keyword_dict['english_terms']:
                        keyword_dict['english_terms'].append('beach')
                    break
                
        print(f"Enhanced keywords: {keyword_dict}")
        
        # Flatten keywords for search
        locations = keyword_dict.get('locations', [])
        english_terms = keyword_dict.get('english_terms', [])
        german_terms = keyword_dict.get('german_terms', [])
        translated_terms = keyword_dict.get('translated_german', [])
        general_terms = keyword_dict.get('general', [])
        
        # All terms for individual searches
        all_keywords = locations + english_terms + german_terms + translated_terms + general_terms
        
        # Search for each keyword separately
        all_results = []
        all_results.extend(original_results)
        
        # Search for specific countries directly in document metadata and content
        # This avoids the $contains error and uses better search methods
        country_docs = []
        
        # Look for Vietnam-related documents directly using the search method
        for country_term in ["Vietnam", "Cambodia", "Laos", "Thailand"]:
            if country_term.lower() in query.lower() or (country_term == "Vietnam" and not any(country.lower() in query.lower() for country in ["Cambodia", "Laos", "Thailand"])):
                try:
                    # Use direct search with the country name
                    print(f"Direct search for documents about '{country_term}'")
                    country_results = self.search(country_term, 10)
                    country_docs.extend(country_results)
                    
                    # Also search for specific cities/locations in the country if mentioned
                    if country_term == "Vietnam":
                        for location in ["Hanoi", "Ho Chi Minh", "Halong Bay", "Hoi An", "Da Nang"]:
                            if location.lower() in query.lower():
                                location_results = self.search(location, 5)
                                country_docs.extend(location_results)
                    
                    print(f"Found {len(country_docs)} documents mentioning '{country_term}'")
                except Exception as e:
                    print(f"Error in country search: {e}")
        
        # Add the direct results
        all_results.extend(country_docs)
        
        # Search for locations first (highest priority)
        for location in locations:
            print(f"Searching with location: '{location}'")
            results = self.search(location, max(2, n_results//2))
            all_results.extend(results)
            
            # Try location with theme terms in both languages
            for term in english_terms + translated_terms:
                combined_query = f"{location} {term}"
                print(f"Searching with: '{combined_query}'")
                results = self.search(combined_query, 2)
                all_results.extend(results)
                
            # Also try with German terms
            for term in german_terms:
                combined_query = f"{location} {term}"
                print(f"Searching with: '{combined_query}' (German)")
                results = self.search(combined_query, 2)
                all_results.extend(results)
        
        # If still need more results, try all terms in both languages
        if len(all_results) < n_results * 2:  # Get extra results for better selection
            # Try English/translated terms
            for term in english_terms + translated_terms:
                print(f"Searching with term: '{term}'")
                results = self.search(term, 1)
                all_results.extend(results)
            
            # Try German terms
            for term in german_terms:
                print(f"Searching with German term: '{term}'")
                results = self.search(term, 1)
                all_results.extend(results)
        
        # Deduplicate results
        unique_results = []
        seen_texts = set()
        
        for result in all_results:
            # Use the first 100 chars as a fingerprint to avoid exact duplicates
            text_start = result['text'][:100] if len(result['text']) >= 100 else result['text']
            if text_start not in seen_texts:
                seen_texts.add(text_start)
                unique_results.append(result)
        
        print(f"Found {len(unique_results)} unique results after deduplication")
        
        # Return the top n results
        return unique_results[:n_results]
        
    def get_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Get relevant context as a concatenated string for a given query.
        
        Args:
            query: The search query
            n_results: Number of results to include
            
        Returns:
            Concatenated string of relevant document chunks
        """
        print("Performing enhanced search to find the most relevant information...")
        results = self.enhanced_search(query, n_results)
        
        if not results:
            print("No relevant information found in the database.")
            return "No relevant information found in the database."
        
        print(f"Found {len(results)} relevant documents.")
        
        # Format results as a single context string
        context_parts = []
        for i, result in enumerate(results):
            source = result['metadata'].get('source', 'Unknown source')
            filename = result['metadata'].get('filename', 'Unknown file')
            # Mark whether this is a MUST-SEE item
            is_must_see = result['metadata'].get('has_must_see', 'no') == 'yes'
            
            # Include more detailed source information and mark RAG database content
            if is_must_see:
                context_parts.append(f"DOCUMENT {i+1} [FROM DATABASE - MUST-SEE] (Source: {source}, File: {filename}):\n{result['text']}\n")
            else:
                context_parts.append(f"DOCUMENT {i+1} [FROM DATABASE] (Source: {source}, File: {filename}):\n{result['text']}\n")
        
        return "\n".join(context_parts)
    
    def get_relevant_context_with_llm_keywords(self, query: str, keywords: dict, n_results: int = 5) -> str:
        """
        Get relevant context using LLM-extracted keywords.
        
        Args:
            query: The original search query
            keywords: Dictionary of keywords extracted by LLM
            n_results: Number of results to include
            
        Returns:
            Concatenated string of relevant document chunks
        """
        import utils  # Import here to avoid circular import
        
        # Use ANSI color codes directly
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        print(f"{CYAN}🔎 Searching with LLM-extracted keywords...{RESET}")
        
        # First collect all MUST-SEE content related to the query
        must_see_results = []
        
        # First, try a direct search with the original query
        original_results = self.search(query, n_results=2)
        
        # Collect MUST-SEE results from original query
        for result in original_results:
            if result['metadata'].get('has_must_see', 'no') == 'yes':
                must_see_results.append(result)
        
        # Perform targeted searches based on extracted keywords
        all_results = []
        all_results.extend(original_results)
        
        # Search by locations (highest priority)
        for location in keywords.get('locations', []):
            print(f"Searching with location: '{location}'")
            location_results = self.search(location, n_results=2)
            
            # Add MUST-SEE content to the special collection
            for result in location_results:
                if result['metadata'].get('has_must_see', 'no') == 'yes':
                    must_see_results.append(result)
                    
            all_results.extend(location_results)
            
            # Combine location with themes
            for theme in keywords.get('themes', []):
                combined_query = f"{location} {theme}"
                print(f"Searching with: '{combined_query}'")
                theme_results = self.search(combined_query, n_results=1)
                all_results.extend(theme_results)
                
                # Add MUST-SEE content to the special collection
                for result in theme_results:
                    if result['metadata'].get('has_must_see', 'no') == 'yes':
                        must_see_results.append(result)
        
        # Search by themes
        for theme in keywords.get('themes', []):
            if theme not in ['beach', 'food']:  # We already prioritized these
                print(f"Searching with theme: '{theme}'")
                theme_results = self.search(theme, n_results=1)
                all_results.extend(theme_results)
        
        # Search by activities
        for activity in keywords.get('activities', []):
            print(f"Searching with activity: '{activity}'")
            activity_results = self.search(activity, n_results=1)
            all_results.extend(activity_results)
        
        # Search by accommodation types
        for accommodation in keywords.get('accommodation_types', []):
            print(f"Searching with accommodation: '{accommodation}'")
            accommodation_results = self.search(accommodation, n_results=1)
            all_results.extend(accommodation_results)
        
        # Ensure MUST-SEE content is in the results
        for result in must_see_results:
            all_results.append(result)
        
        # Deduplicate results
        unique_results = []
        seen_texts = set()
        
        # First add all MUST-SEE content to ensure they're included
        for result in must_see_results:
            text_start = result['text'][:100] if len(result['text']) >= 100 else result['text']
            if text_start not in seen_texts:
                seen_texts.add(text_start)
                unique_results.append(result)
        
        # Then add other results
        for result in all_results:
            text_start = result['text'][:100] if len(result['text']) >= 100 else result['text']
            if text_start not in seen_texts:
                seen_texts.add(text_start)
                unique_results.append(result)
        
        print(f"{GREEN}✓ Found {len(unique_results)} unique results (including {len(must_see_results)} MUST-SEE items){RESET}")
        
        # If no results found, fall back to regular search
        if not unique_results:
            print(f"{YELLOW}⚠ No results from keyword search, falling back to enhanced search...{RESET}")
            return self.get_relevant_context(query, n_results)
        
        # Check if context optimization is enabled in config
        if hasattr(self.config, 'ENABLE_CONTEXT_OPTIMIZATION') and self.config.ENABLE_CONTEXT_OPTIMIZATION:
            # Optimize context by extracting only the most relevant parts
            optimized_context, original_tokens, optimized_tokens = self.optimize_context(unique_results, query, keywords)
            
            # Check if the optimized context is below the token limit
            if optimized_tokens > self.config.MAX_CONTEXT_TOKENS:
                print(f"{YELLOW}⚠ Optimized context ({optimized_tokens:,} tokens) exceeds the maximum token limit ({self.config.MAX_CONTEXT_TOKENS:,} tokens).{RESET}")
                print(f"{YELLOW}⚠ Further reducing context to fit within token limit...{RESET}")
                
                # Calculate how much we need to reduce
                reduction_factor = self.config.MAX_CONTEXT_TOKENS / optimized_tokens
                
                # For smaller chunks, we need to be more aggressive with reduction
                # Apply a more aggressive reduction factor for smaller chunks
                adjusted_max_words = int(self.config.MAX_WORDS_PER_DOCUMENT * reduction_factor * 0.8)
                
                # Ensure we don't go below a reasonable minimum
                adjusted_max_words = max(30, adjusted_max_words)
                
                print(f"{CYAN}ℹ️ Adjusting maximum words per document from {self.config.MAX_WORDS_PER_DOCUMENT} to {adjusted_max_words}{RESET}")
                
                # Save original config
                temp_config = self.config.MAX_WORDS_PER_DOCUMENT
                self.config.MAX_WORDS_PER_DOCUMENT = adjusted_max_words
                
                # Try optimizing again with reduced word count
                optimized_context, original_tokens, optimized_tokens = self.optimize_context(unique_results, query, keywords)
                
                # Restore original config
                self.config.MAX_WORDS_PER_DOCUMENT = temp_config
            
            # Record tokens saved in token tracker if available
            tokens_saved = original_tokens - optimized_tokens
            
            # If the agent has a token tracker, record the saved tokens
            # To avoid circular imports, we need to access this via the config
            if hasattr(self, 'agent') and hasattr(self.agent, 'token_tracker'):
                self.agent.token_tracker.record_tokens_saved(tokens_saved)
            elif hasattr(self.config, '_token_tracker'):
                self.config._token_tracker.record_tokens_saved(tokens_saved)
            
            return optimized_context
        else:
            # If optimization is disabled, use the traditional approach
            print(f"{YELLOW}Context optimization is disabled. Using full document content.{RESET}")
            
            # Format results as a single context string
            context_parts = []
            
            # First add a section for MUST-SEE content if any was found
            must_see_parts = []
            for i, result in enumerate(must_see_results):
                source = result['metadata'].get('source', 'Unknown source')
                filename = result['metadata'].get('filename', 'Unknown file')
                must_see_parts.append(f"MUST-SEE ITEM {i+1} [FROM DATABASE - MUST-SEE] (Source: {source}, File: {filename}):\n{result['text']}\n")
                
            if must_see_parts:
                context_parts.append("## IMPORTANT TRAVEL HIGHLIGHTS [FROM DATABASE]\n\n" + "\n".join(must_see_parts))
            
            # Then add the rest of the results
            general_parts = []
            for i, result in enumerate(unique_results[:n_results]):  # Limit to requested number
                # Skip if this is already in must_see_parts to avoid duplication
                text_start = result['text'][:100] if len(result['text']) >= 100 else result['text']
                if any(text_start in part for part in must_see_parts):
                    continue
                    
                source = result['metadata'].get('source', 'Unknown source')
                filename = result['metadata'].get('filename', 'Unknown file')
                
                # Mark content as from the RAG database
                general_parts.append(f"DOCUMENT {i+1} [FROM DATABASE] (Source: {source}, File: {filename}):\n{result['text']}\n")
            
            if general_parts:
                context_parts.append("## GENERAL TRAVEL INFORMATION [FROM DATABASE]\n\n" + "\n".join(general_parts))
            
            return "\n".join(context_parts)
        
    def get_source_documents(self, query: str, n_results: int = 5) -> list:
        """
        Get a list of source documents used for a given query.
        
        Args:
            query: The search query
            n_results: Number of results to include
            
        Returns:
            List of source documents
        """
        # Use enhanced search for better results
        results = self.enhanced_search(query, n_results)
        
        if not results:
            return []
        
        # Extract and format sources
        sources = []
        for result in results:
            source = result['metadata'].get('source', 'Unknown source')
            filename = result['metadata'].get('filename', 'Unknown file')
            
            # Create a source entry that wasn't already added
            source_entry = f"{filename}"
            if source_entry not in sources:
                sources.append(source_entry)
        
        return sources
    
    def get_source_documents_with_llm_keywords(self, query: str, keywords: dict, n_results: int = 5) -> list:
        """
        Get a list of source documents using LLM-extracted keywords.
        
        Args:
            query: The original search query
            keywords: Dictionary of keywords extracted by LLM
            n_results: Number of results to include
            
        Returns:
            List of source documents
        """
        # First get relevant context with keywords
        context = self.get_relevant_context_with_llm_keywords(query, keywords, n_results)
        
        # If no context, return empty list
        if context == "No relevant information found in the database.":
            return []
        
        # Extract document sources from the context
        sources = []
        for line in context.split("\n"):
            if line.startswith("DOCUMENT") and "File:" in line:
                # Extract filename from context line
                filename_match = re.search(r"File: ([^)]+)", line)
                if filename_match:
                    filename = filename_match.group(1).strip()
                    if filename not in sources:
                        sources.append(filename)
        
        return sources
        
    def optimize_context(self, documents: list, query: str, keywords: dict) -> tuple:
        """
        Optimize the context by extracting only the most relevant parts of each document.
        
        Args:
            documents: List of document dictionaries
            query: Original user query
            keywords: Dictionary of keywords extracted by LLM
            
        Returns:
            Tuple of (optimized_context, original_token_count, optimized_token_count)
        """
        import utils  # Import here to avoid circular import
        
        # Use ANSI color codes directly
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        print(f"\n{CYAN}🔍 Optimizing RAG context for more efficient processing...{RESET}")
        
        # Collect all extracted keywords for relevance scoring
        all_keywords = []
        
        # Add locations (highest priority)
        if 'locations' in keywords:
            all_keywords.extend([(location.lower(), 3.0) for location in keywords.get('locations', [])])
            
        # Add themes (medium-high priority)
        if 'themes' in keywords:
            all_keywords.extend([(theme.lower(), 2.0) for theme in keywords.get('themes', [])])
            
        # Add activities (medium priority)
        if 'activities' in keywords:
            all_keywords.extend([(activity.lower(), 1.5) for activity in keywords.get('activities', [])])
            
        # Add accommodation types (medium-low priority)
        if 'accommodation_types' in keywords:
            all_keywords.extend([(acc_type.lower(), 1.2) for acc_type in keywords.get('accommodation_types', [])])
        
        # Add general keywords (low priority)
        if 'general' in keywords:
            all_keywords.extend([(keyword.lower(), 1.0) for keyword in keywords.get('general', [])])
            
        # Process each document and extract the most relevant paragraphs
        optimized_documents = []
        original_word_count = 0
        optimized_word_count = 0
        
        for doc in documents:
            original_text = doc['text']
            original_word_count += len(original_text.split())
            
            # Split document into paragraphs
            paragraphs = [p.strip() for p in original_text.split('\n\n') if p.strip()]
            
            # If it's not many paragraphs already, try splitting by newlines
            if len(paragraphs) <= 2:
                paragraphs = [p.strip() for p in original_text.split('\n') if p.strip()]
            
            # Score each paragraph based on keyword relevance
            scored_paragraphs = []
            
            for paragraph in paragraphs:
                if not paragraph or len(paragraph.split()) < 5:  # Skip very short paragraphs
                    continue
                    
                # Convert to lowercase for matching
                paragraph_lower = paragraph.lower()
                
                # Initialize score
                score = 0.0
                
                # Check for MUST-SEE content (highest priority)
                if "must-see" in paragraph_lower or "must see" in paragraph_lower:
                    score += 5.0
                    
                # Score based on keywords
                for keyword, weight in all_keywords:
                    if keyword in paragraph_lower:
                        # Add the weight for each occurrence of the keyword
                        occurrences = paragraph_lower.count(keyword)
                        score += weight * occurrences
                
                # Bonus for first paragraph (often contains summary information)
                if paragraph == paragraphs[0]:
                    score += 1.0
                    
                # Small bonus for paragraphs with reasonable length (adjusted for smaller chunks)
                words = len(paragraph.split())
                if 10 <= words <= 50:
                    score += 0.8  # Higher bonus for appropriately sized paragraphs
                    
                scored_paragraphs.append((paragraph, score, words))
            
            # Sort paragraphs by score in descending order
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate target word count for this document
            max_words = self.config.MAX_WORDS_PER_DOCUMENT
            
            # Select top paragraphs up to the word limit
            selected_paragraphs = []
            current_word_count = 0
            
            for paragraph, score, words in scored_paragraphs:
                if current_word_count + words <= max_words or not selected_paragraphs:
                    selected_paragraphs.append(paragraph)
                    current_word_count += words
                else:
                    # If we can't add more paragraphs, break the loop
                    break
            
            # If the document has MUST-SEE content, make sure it's explicitly marked
            has_must_see = doc['metadata'].get('has_must_see', 'no') == 'yes'
            
            # Create optimized text by joining selected paragraphs
            optimized_text = "\n\n".join(selected_paragraphs)
            
            # Add a summary line if the text was significantly reduced
            original_tokens = utils.estimate_tokens(original_text)
            optimized_tokens = utils.estimate_tokens(optimized_text)
            
            if has_must_see:
                # Add a clear marker for MUST-SEE content at the beginning
                optimized_text = f"🌟 MUST-SEE CONTENT 🌟\n\n{optimized_text}"
                
                # Also add a special tag for the LLM to recognize in the plan
                optimized_text += "\n\n[THIS IS MUST-SEE CONTENT THAT SHOULD BE TAGGED WITH [FROM DATABASE - MUST-SEE] IN THE PLAN]"
            
            # If the document was significantly reduced, add a note
            if original_tokens > optimized_tokens * 1.5:
                reduction = (1 - (optimized_tokens / original_tokens)) * 100
                optimized_text += f"\n\n[Note: This is an optimized extract (reduced by {reduction:.0f}%) from a larger document]"
            
            # Create optimized document with original metadata
            optimized_doc = {
                'text': optimized_text,
                'metadata': doc['metadata'],
                'score': doc.get('score', 0),
                'original_tokens': original_tokens,
                'optimized_tokens': optimized_tokens
            }
            
            optimized_documents.append(optimized_doc)
            optimized_word_count += len(optimized_text.split())
            
            # Print progress with colored output
            if has_must_see:
                print(f"{YELLOW}🌟 Optimized MUST-SEE document: {doc['metadata'].get('filename', 'Unknown')}{RESET}")
            else:
                print(f"{GREEN}✓ Optimized document: {doc['metadata'].get('filename', 'Unknown')}{RESET}")
                
            reduction_pct = (1 - (optimized_tokens / original_tokens)) * 100 if original_tokens > 0 else 0
            print(f"  - Reduced from {original_tokens:,} to {optimized_tokens:,} tokens ({reduction_pct:.1f}% saving)")
        
        # Calculate token counts
        original_token_count = utils.estimate_tokens("\n\n".join([doc['text'] for doc in documents]))
        
        # Format optimized context as a single string
        optimized_context_parts = []
        
        # First add a section for MUST-SEE content if any was found
        must_see_parts = []
        for i, doc in enumerate([d for d in optimized_documents if d['metadata'].get('has_must_see', 'no') == 'yes']):
            source = doc['metadata'].get('source', 'Unknown source')
            filename = doc['metadata'].get('filename', 'Unknown file')
            must_see_parts.append(f"MUST-SEE ITEM {i+1} [FROM DATABASE - MUST-SEE] (Source: {source}, File: {filename}):\n{doc['text']}\n")
            
        if must_see_parts:
            optimized_context_parts.append("## IMPORTANT TRAVEL HIGHLIGHTS [FROM DATABASE]\n\n" + "\n".join(must_see_parts))
        
        # Then add the rest of the results
        general_parts = []
        for i, doc in enumerate([d for d in optimized_documents if d['metadata'].get('has_must_see', 'no') != 'yes']):
            # Skip if this is already in must_see_parts to avoid duplication
            if any(doc['text'][:100] in part for part in must_see_parts):
                continue
                
            source = doc['metadata'].get('source', 'Unknown source')
            filename = doc['metadata'].get('filename', 'Unknown file')
            
            # Mark content as from the RAG database
            general_parts.append(f"DOCUMENT {i+1} [FROM DATABASE] (Source: {source}, File: {filename}):\n{doc['text']}\n\n[THIS INFORMATION SHOULD BE TAGGED WITH [FROM DATABASE] WHEN USED IN THE PLAN]\n")
        
        if general_parts:
            optimized_context_parts.append("## GENERAL TRAVEL INFORMATION [FROM DATABASE]\n\n" + "\n".join(general_parts))
        
        optimized_context = "\n".join(optimized_context_parts)
        optimized_token_count = utils.estimate_tokens(optimized_context)
        
        # Calculate and report savings
        tokens_saved = original_token_count - optimized_token_count
        tokens_saved_pct = (tokens_saved / original_token_count) * 100 if original_token_count > 0 else 0
        
        print(f"\n{CYAN}📊 RAG Context Optimization Summary:{RESET}")
        print(f"  - Original documents: {len(documents)}")
        print(f"  - Original tokens: {original_token_count:,}")
        print(f"  - Optimized tokens: {optimized_token_count:,}")
        print(f"  - Tokens saved: {tokens_saved:,} ({tokens_saved_pct:.1f}%)")
        print(f"  - Original word count: {original_word_count:,}")
        print(f"  - Optimized word count: {optimized_word_count:,}")
        print(f"  - Word reduction: {original_word_count - optimized_word_count:,} words ({(original_word_count - optimized_word_count) / original_word_count * 100:.1f}%)\n")
        
        return optimized_context, original_token_count, optimized_token_count