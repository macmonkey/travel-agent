#!/usr/bin/env python3
"""
Main entry point for the Travel Plan Agent application.
This script initializes the agent and handles user interaction.
"""

import os
import re
import time
import sys
import logging
import argparse
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from rag_database import RAGDatabase
from agent import TravelAgent
from utils import save_travel_plan

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_with_typing_effect(text, speed=0.01, color=None):
    """Print text with a typing effect."""
    if color:
        sys.stdout.write(color)

    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)

    if color:
        sys.stdout.write(Colors.END)

    sys.stdout.write('\n')

def print_logo():
    """Print the app logo."""
    logo = """
    ╭────────────────────────────────────────────────╮
    │                                                │
    │            🌴 REISEPLAN AGENT 🌴               │
    │         Personalisierte Reiseplanung          │
    │               mit KI-Assistenz                │
    │                                                │
    ╰────────────────────────────────────────────────╯
    """
    print(Colors.CYAN + logo + Colors.END)

def print_section_header(text):
    """Print a section header."""
    print("\n" + Colors.BOLD + Colors.YELLOW + "== " + text + " ==" + Colors.END)

def ask_user(prompt):
    """Ask the user a question with styling."""
    print(Colors.GREEN + "\n➤ " + prompt + Colors.END)
    return input("  ")

def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description="Travel Plan Agent - Generate personalized travel plans with AI")
    
    # Add argument groups for better organization
    cache_group = parser.add_argument_group('Cache Management')
    rate_group = parser.add_argument_group('Rate Limiting')
    general_group = parser.add_argument_group('General Options')
    
    # Cache management options
    cache_group.add_argument('--clear-cache', action='store_true', 
                            help='Clear the response cache completely')
    cache_group.add_argument('--clear-old-cache', type=int, metavar='DAYS',
                            help='Clear cache entries older than specified days')
    cache_group.add_argument('--no-cache', action='store_true',
                            help='Disable response caching')
    
    # Rate limiting options
    rate_group.add_argument('--min-delay', type=float, metavar='SECONDS',
                           help='Minimum delay between API calls (default: 3.0)')
    rate_group.add_argument('--batch-size', type=int, metavar='NUM',
                           help='Number of requests before taking a longer pause (default: 5)')
    rate_group.add_argument('--batch-pause', type=float, metavar='SECONDS',
                           help='Seconds to pause after batch_size requests (default: 15.0)')
    
    # General options
    general_group.add_argument('--reindex', action='store_true',
                              help='Force reindexing of documents')
    general_group.add_argument('--info', action='store_true',
                              help='Show information about the agent and exit')
    
    # API usage optimization
    api_group = parser.add_argument_group('API Usage Optimization')
    api_group.add_argument('--fast-mode', action='store_true',
                          help='Enable fast mode (skips metadata generation to reduce API calls)')
    api_group.add_argument('--ultra-fast-mode', action='store_true',
                          help='Enable ultra-fast mode (optimizes all possible API calls)')
    api_group.add_argument('--direct-mode', action='store_true',
                          help='Skip draft plan and go directly to detailed plan')
    api_group.add_argument('--optimize-keywords', action='store_true',
                          help='Skip second keyword extraction API call')
    api_group.add_argument('--minimal-mode', action='store_true',
                          help='Use absolute minimal API calls (2 calls only)')
    api_group.add_argument('--offline-mode', action='store_true',
                          help='Skip ALL rate limiting and delays (dangerous, use only when you have private API)')
    api_group.add_argument('--api-quota', type=int, metavar='NUM',
                          help='Set approximate number of API calls allowed (reduces features for lower values)')
    
    return parser.parse_args()

def main():
    """Main function to run the Travel Plan Agent application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("travel_agent.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("TravelAgent")

    # Print logo
    print_logo()
    print_with_typing_effect("Initializing your personalized travel planning assistant...",
                         speed=0.01, color=Colors.CYAN)

    # Initialize configuration
    config = Config()
    logger.info("Configuration loaded")

    # Initialize document processor
    doc_processor = DocumentProcessor(config)

    # Initialize RAG database
    rag_db = RAGDatabase(config)

    # Initialize travel agent with the RAG database
    agent = TravelAgent(config, rag_db)
    
    # Handle cache-related command line arguments
    if args.clear_cache:
        print_section_header("Cache Management")
        print_with_typing_effect("Clearing response cache completely...", speed=0.01)
        removed = agent.clear_cache()
        print_with_typing_effect(f"Removed {removed} cache entries", speed=0.01, color=Colors.GREEN)
        if not args.info:  # If just clearing cache without info flag, exit
            return
    
    if args.clear_old_cache:
        print_section_header("Cache Management")
        days = args.clear_old_cache
        print_with_typing_effect(f"Clearing cache entries older than {days} days...", speed=0.01)
        removed = agent.clear_cache(older_than_days=days)
        print_with_typing_effect(f"Removed {removed} cache entries", speed=0.01, color=Colors.GREEN)
        if not args.info:  # If just clearing cache without info flag, exit
            return
    
    if args.no_cache:
        print_with_typing_effect("Response caching is disabled", speed=0.01)
        agent.cache_enabled = False
    
    # Apply rate limiting settings if provided
    if args.min_delay:
        agent.min_delay_between_calls = args.min_delay
        print_with_typing_effect(f"Setting minimum delay between API calls to {args.min_delay}s", speed=0.01)
    
    if args.batch_size:
        agent.batch_size = args.batch_size
        print_with_typing_effect(f"Setting batch size to {args.batch_size} requests", speed=0.01)
    
    if args.batch_pause:
        agent.batch_pause = args.batch_pause
        print_with_typing_effect(f"Setting batch pause to {args.batch_pause}s", speed=0.01)
        
    # Handle API optimization options
    if args.offline_mode:
        # Offline mode - aggressively skip all rate limiting
        config.OFFLINE_MODE = True
        config.SKIP_METADATA_GENERATION = True
        config.OPTIMIZE_KEYWORD_EXTRACTION = True
        config.SKIP_DRAFT_PLAN = True
        print_with_typing_effect("⚠️ OFFLINE MODE ENABLED: Skipping ALL rate limiting", 
                             speed=0.01, color=Colors.RED + Colors.BOLD)
        print_with_typing_effect("    - DANGEROUS: May result in API rate limit bans", speed=0.01, color=Colors.RED)
        print_with_typing_effect("    - Use only if you have a private API key with high quotas", speed=0.01, color=Colors.RED)
        print_with_typing_effect("    - All delays and rate limiting disabled", speed=0.01, color=Colors.RED)
        print_with_typing_effect("    - Fast but risky operation", speed=0.01, color=Colors.RED)
    elif args.minimal_mode:
        # Absolute minimal mode - just 2 API calls
        config.SKIP_METADATA_GENERATION = True
        config.OPTIMIZE_KEYWORD_EXTRACTION = True
        config.SKIP_DRAFT_PLAN = True
        print_with_typing_effect("⚡⚡⚡ Minimal mode enabled: Absolute minimal API usage", 
                             speed=0.01, color=Colors.RED + Colors.BOLD)
        print_with_typing_effect("    - Skipping draft plan generation", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - Skipping metadata generation", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - Using rule-based keyword extraction", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - API calls reduced from 5 to 2", speed=0.01, color=Colors.YELLOW)
        # Adjust delays for minimal mode
        agent.min_delay_between_calls = 3.0
        agent.batch_size = 2
        agent.batch_pause = 15.0
    elif args.ultra_fast_mode:
        # Ultra fast mode enables most optimizations
        config.SKIP_METADATA_GENERATION = True
        config.OPTIMIZE_KEYWORD_EXTRACTION = True
        print_with_typing_effect("⚡⚡ Ultra-fast mode enabled: Using strong API optimizations", 
                             speed=0.01, color=Colors.YELLOW + Colors.BOLD)
        print_with_typing_effect("    - Skipping metadata generation", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - Using rule-based keyword extraction for feedback", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - API calls reduced from 5 to 3", speed=0.01, color=Colors.YELLOW)
    elif args.fast_mode:
        config.SKIP_METADATA_GENERATION = True
        print_with_typing_effect("⚡ Fast mode enabled: Skipping metadata generation to reduce API calls", 
                             speed=0.01, color=Colors.YELLOW + Colors.BOLD)
    
    if args.direct_mode and not args.minimal_mode:
        config.SKIP_DRAFT_PLAN = True
        print_with_typing_effect("🚀 Direct mode enabled: Skipping draft plan and going directly to detailed plan", 
                             speed=0.01, color=Colors.YELLOW + Colors.BOLD)
        print_with_typing_effect("    - API calls reduced by skipping draft plan step", speed=0.01, color=Colors.YELLOW)
    
    if args.optimize_keywords and not args.ultra_fast_mode and not args.minimal_mode:
        config.OPTIMIZE_KEYWORD_EXTRACTION = True
        print_with_typing_effect("🔑 Optimized keyword extraction: Using rule-based analysis for feedback", 
                             speed=0.01, color=Colors.YELLOW)
    
    if args.api_quota:
        if args.api_quota < 3:
            print_with_typing_effect("⚠️ Extremely low API quota. Only minimum functionality possible.", 
                               speed=0.01, color=Colors.RED)
            config.SKIP_METADATA_GENERATION = True
            config.OPTIMIZE_KEYWORD_EXTRACTION = True
            agent.min_delay_between_calls = 10.0  # Increase delay between calls significantly
            print_with_typing_effect("    - Minimum features: only draft plan generation possible", speed=0.01, color=Colors.RED)
        elif args.api_quota < 4:
            print_with_typing_effect("⚠️ Very low API quota set. Using ultra-fast mode.", 
                               speed=0.01, color=Colors.RED)
            config.SKIP_METADATA_GENERATION = True
            config.OPTIMIZE_KEYWORD_EXTRACTION = True
            agent.min_delay_between_calls = 5.0  # Increase delay between calls
        elif args.api_quota < 5:
            print_with_typing_effect("ℹ️ Low API quota set. Using optimized keyword extraction.", 
                               speed=0.01, color=Colors.YELLOW)
            config.SKIP_METADATA_GENERATION = True
            config.OPTIMIZE_KEYWORD_EXTRACTION = True
        elif args.api_quota < 10:
            print_with_typing_effect("ℹ️ Limited API quota set. Skipping metadata generation.", 
                               speed=0.01, color=Colors.YELLOW)
            config.SKIP_METADATA_GENERATION = True
        print_with_typing_effect(f"API quota set to approximately {args.api_quota} calls", speed=0.01)
    
    # Show agent info if requested
    if args.info:
        print_section_header("Agent Information")
        print_with_typing_effect(f"Gemini Model: {config.GEMINI_MODEL}", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Rate Limiting: {agent.min_delay_between_calls}s between calls", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Batch Processing: {agent.batch_size} requests then {agent.batch_pause}s pause", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Caching: {'Enabled' if agent.cache_enabled else 'Disabled'}", speed=0.01, color=Colors.CYAN)
        
        # Count cache entries
        if agent.cache_enabled:
            cache_count = len(os.listdir(agent.cache_dir)) - 1  # Subtract 1 for config file
            if cache_count > 0:
                print_with_typing_effect(f"Cache entries: {cache_count}", speed=0.01, color=Colors.CYAN)
        
        # Show optimization settings
        print_with_typing_effect("\nAPI Usage Optimization:", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect(f"  Skip draft plan: {'Enabled' if config.SKIP_DRAFT_PLAN else 'Disabled'}", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect(f"  Skip metadata generation: {'Enabled' if config.SKIP_METADATA_GENERATION else 'Disabled'}", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect(f"  Optimize keyword extraction: {'Enabled' if config.OPTIMIZE_KEYWORD_EXTRACTION else 'Disabled'}", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect(f"  Use cached keywords: {'Enabled' if config.USE_CACHED_KEYWORDS else 'Disabled'}", 
                            speed=0.01, color=Colors.YELLOW)
        
        if hasattr(config, 'OFFLINE_MODE') and config.OFFLINE_MODE:
            print_with_typing_effect(f"  OFFLINE MODE: {'Enabled' if config.OFFLINE_MODE else 'Disabled'}", 
                                speed=0.01, color=Colors.RED + Colors.BOLD)
        
        print_with_typing_effect("\nTypical API calls per plan:", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Standard mode: 5 calls (2× keywords, draft, detailed, metadata)", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Fast mode: 4 calls (no metadata generation)", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Direct mode: 4 calls (no draft plan)", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Optimized keywords: 4 calls (1× keywords, draft, detailed, metadata)", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Ultra-fast mode: 3 calls (1× keywords, draft, detailed)", 
                            speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("  - Minimal mode: 2 calls (1× keywords, detailed only)", 
                            speed=0.01, color=Colors.RED)
        print_with_typing_effect("  - Offline mode: 2 calls (same as minimal, but NO rate limiting)", 
                            speed=0.01, color=Colors.RED)
        
        print()
        return
    
    # Force reindexing if requested
    force_reindex = args.reindex
    
    # Delete existing ChromaDB directory to ensure a clean start
    if force_reindex and os.path.exists(config.CHROMA_DB_PATH):
        print_with_typing_effect("Removing existing database for clean reindex...", speed=0.01)
        import shutil
        try:
            # First try to fix permissions on the directory
            try:
                os.system(f"chmod -R 777 {config.CHROMA_DB_PATH}")
                print_with_typing_effect("Updated permissions on database directory", speed=0.01)
            except:
                pass
                
            shutil.rmtree(config.CHROMA_DB_PATH)
            config.CHROMA_DB_PATH.mkdir(exist_ok=True)
            # Ensure permissions are set properly
            os.system(f"chmod -R 777 {config.CHROMA_DB_PATH}")
            print_with_typing_effect("Database directory cleared for fresh indexing", speed=0.01)
        except Exception as e:
            print(f"Error clearing database directory: {e}")

    # Check if database needs to be initialized with documents
    if force_reindex or not os.path.exists(config.CHROMA_DB_PATH) or len(os.listdir(config.CHROMA_DB_PATH)) == 0:
        print_section_header("Database Initialization")
        print_with_typing_effect("Initializing database with travel documents...", speed=0.02)
        documents = doc_processor.process_directory(config.DOCUMENTS_PATH)
        print_with_typing_effect(f"Processing {len(documents)} documents...", speed=0.02)

        # Count document chunks
        total_chunks = 0
        for doc in documents:
            if 'text' in doc and doc['text']:
                total_chunks += 1

        print_with_typing_effect(f"Created {total_chunks} text chunks from {len(documents)} documents",
                             speed=0.02, color=Colors.CYAN)

        # Proceed with indexing if documents were found
        if documents:
            rag_db.index_documents(documents)
            print_with_typing_effect(f"Indexed {len(documents)} documents into the database.",
                                 speed=0.02, color=Colors.GREEN)
        else:
            print_with_typing_effect("No documents found to index!",
                                 speed=0.02, color=Colors.RED)

    # Diagnostic information about the database
    print_section_header("Database Information")
    collection_info = rag_db.get_collection_info()
    print_with_typing_effect(f"Database contains {collection_info['count']} documents",
                         speed=0.01, color=Colors.CYAN)
    print_with_typing_effect(f"Collection name: {collection_info['name']}",
                         speed=0.01, color=Colors.CYAN)

    # Create output directory for saved plans
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print_section_header("Welcome")
    print_with_typing_effect("I'm your travel planning assistant!", speed=0.02)
    print_with_typing_effect("Tell me about your dream vacation, and I'll create a personalized travel plan.", speed=0.02)
    print_with_typing_effect("You can type 'quit' or 'exit' at any time to end our session.", speed=0.02)

    while True:
        # Get user input
        user_query = ask_user("What kind of trip are you planning?")

        if user_query.lower() in ["quit", "exit"]:
            print_with_typing_effect("Thank you for using the Travel Plan Agent. Goodbye!",
                                 speed=0.02, color=Colors.CYAN)
            break

        # Process user query and generate travel plan
        try:
            # Check if we should skip the draft plan
            if config.SKIP_DRAFT_PLAN:
                print_section_header("Direct Plan Creation")
                print_with_typing_effect("Skipping draft plan and going directly to detailed plan...", 
                                     speed=0.02, color=Colors.YELLOW)
                
                # Set an empty draft plan
                draft_plan = ""
                # No feedback needed since we're skipping draft
                feedback = "Create a detailed plan directly."
            else:
                # Generate draft plan
                print_section_header("Creating Draft Plan")
                print_with_typing_effect("Analyzing your preferences and searching for relevant information...", speed=0.02)
                draft_plan = agent.generate_draft_plan(user_query)

                print_section_header("Draft Travel Plan")
                print(Colors.CYAN + draft_plan + Colors.END)

                # Ask for user feedback
                feedback = ask_user("Do you like this plan? Any changes or specific requests?")

            # Continue if user wants to proceed (or if we're using direct mode)
            if feedback.lower() not in ["no", "quit", "exit"] or config.SKIP_DRAFT_PLAN:
                # Generate detailed plan with user feedback
                print_section_header("Creating Detailed Plan")
                print_with_typing_effect("Incorporating your feedback and creating a detailed itinerary...", speed=0.02)

                # Show search process in real-time (handled by RAG database logging)
                print_section_header("RAG Search Process")
                print_with_typing_effect("Searching knowledge base for relevant information...", speed=0.01)
                time.sleep(0.5)  # Small pause to let the search happen and show logs

                # First part - generate the actual plan
                print_section_header("Creating Comprehensive Plan")
                print_with_typing_effect("Step 1/2: Creating comprehensive travel plan...", speed=0.01)
                detailed_plan = agent.generate_detailed_plan(user_query, draft_plan, feedback)

                print_section_header("Detailed Travel Plan")
                print(Colors.CYAN + detailed_plan + Colors.END)

                # Save the plan with its title - the save_travel_plan function will extract the title
                print_with_typing_effect("Saving your travel plan...", speed=0.02)
                saved_files = save_travel_plan(detailed_plan)

                print_section_header("Plan Saved")
                # Get just the filename part, not the full path, for prettier display
                md_filename = os.path.basename(saved_files['markdown']) if 'markdown' in saved_files else "No markdown saved"
                txt_filename = os.path.basename(saved_files['txt']) if 'txt' in saved_files else "No text saved"
                
                # Display a nice summary of the saved plan
                if 'title' in saved_files:
                    print_with_typing_effect(f"✅ Created: {saved_files['title']}", speed=0.01, color=Colors.CYAN + Colors.BOLD)
                
                if 'countries' in saved_files and saved_files['countries']:
                    countries_str = ", ".join(saved_files['countries'])
                    print_with_typing_effect(f"🌍 Destinations: {countries_str}", speed=0.01, color=Colors.YELLOW)
                
                if 'days' in saved_files and saved_files['days']:
                    print_with_typing_effect(f"⏱️ Duration: {saved_files['days']} days", speed=0.01, color=Colors.YELLOW)
                
                print_with_typing_effect(f"📄 Files:", speed=0.01, color=Colors.GREEN)
                print_with_typing_effect(f"   - {md_filename}", speed=0.01, color=Colors.GREEN)
                print_with_typing_effect(f"   - {txt_filename}", speed=0.01, color=Colors.GREEN)
                
                print_with_typing_effect("Your travel plan has been saved to the output directory.",
                                     speed=0.01, color=Colors.CYAN)
            else:
                print_with_typing_effect("Operation cancelled. Let's try again!",
                                     speed=0.02, color=Colors.YELLOW)

        except Exception as e:
            logger.error(f"Error generating travel plan: {str(e)}", exc_info=True)
            print_with_typing_effect(f"An error occurred: {str(e)}",
                                 speed=0.02, color=Colors.RED)

if __name__ == "__main__":
    main()
