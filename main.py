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
# Import the new Milvus database module
try:
    from milvus_database import MilvusDatabase
except ImportError:
    print("Milvus database module not available. Install pymilvus to use Milvus/Zilliz.")
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
    input_group = parser.add_argument_group('Input Options')
    db_group = parser.add_argument_group('Database Management')

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

    # Input options
    input_group.add_argument('--prompt', type=str, metavar='FILE',
                           help='Path to a text file containing the prompt to use')
    input_group.add_argument('--prompt-text', type=str, metavar='TEXT',
                           help='Directly specify the prompt text to use')

    # Database management options
    db_group.add_argument('--use-milvus', action='store_true',
                         help='Use Milvus/Zilliz Cloud instead of local ChromaDB')
    db_group.add_argument('--initial-index', action='store_true',
                         help='Perform initial indexing of all documents (use at first setup or for new documents)')
    db_group.add_argument('--index-status', action='store_true',
                         help='Show detailed information about the current index status and exit')
    db_group.add_argument('--reindex-file', type=str, metavar='FILE',
                         help='Reindex a specific file or glob pattern of files')
    db_group.add_argument('--recreate-index', action='store_true',
                         help='Completely recreate the database index from scratch')
    db_group.add_argument('--sync-documents', action='store_true',
                         help='Synchronize local documents with the database (index new/changed files only)')
    db_group.add_argument('--export-index', type=str, metavar='FILE',
                         help='Export index metadata to a file for backup')
    db_group.add_argument('--import-index', type=str, metavar='FILE',
                         help='Import index metadata from a file')

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
    api_group.add_argument('--force-mode', action='store_true',
                          help='Skip ALL rate limiting and delays for maximum speed (use with caution)')
    api_group.add_argument('--chat-mode', action='store_true',
                          help='Use chat-based plan generation (single API session, greatly reduces rate limit issues)')
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

    # Check if Milvus is requested
    if args.use_milvus:
        config.USE_MILVUS = True
        print_with_typing_effect("Using Milvus/Zilliz Cloud for vector database", speed=0.01, color=Colors.CYAN)
        
        # Verify token is available
        if not config.MILVUS_TOKEN:
            print_with_typing_effect("Error: No Milvus API token provided", speed=0.01, color=Colors.RED)
            print_with_typing_effect("Please set the MILVUS_TOKEN environment variable in .env file", speed=0.01, color=Colors.RED)
            print_with_typing_effect("Falling back to ChromaDB...", speed=0.01, color=Colors.YELLOW)
            config.USE_MILVUS = False
            rag_db = RAGDatabase(config)
        else:
            try:
                # Initialize Milvus database with a timeout to prevent hanging
                os.environ["PYMILVUS_CONNECTION_TIMEOUT"] = "10"  # 10 seconds timeout
                print_with_typing_effect(f"Connecting to: {config.MILVUS_URI}", speed=0.01, color=Colors.CYAN)
                rag_db = MilvusDatabase(config, force_recreate=args.recreate_index)
                print_with_typing_effect("Connected to Milvus/Zilliz Cloud successfully", speed=0.01, color=Colors.GREEN)
            except Exception as e:
                print_with_typing_effect(f"Error connecting to Milvus/Zilliz: {e}", speed=0.01, color=Colors.RED)
                print_with_typing_effect("Falling back to ChromaDB...", speed=0.01, color=Colors.YELLOW)
                config.USE_MILVUS = False
                rag_db = RAGDatabase(config)
    else:
        # Initialize ChromaDB database
        rag_db = RAGDatabase(config)
        
    # Handle database commands
    if args.index_status:
        print_section_header("Database Status")
        
        if config.USE_MILVUS:
            # Get detailed index status from Milvus
            status = rag_db.get_index_status()
            
            print_with_typing_effect(f"Database Type: {status['database_type']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Server: {status['server']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Collection: {status['collection_name']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Last Indexed: {status['last_indexed']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Indexed Files: {status['total_files']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Total Chunks: {status['total_chunks']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Avg Chunks/File: {status['avg_chunks_per_file']:.1f}", speed=0.01, color=Colors.CYAN)
            
            print_with_typing_effect("\nFile Types:", speed=0.01, color=Colors.YELLOW)
            for ext, count in status['file_types'].items():
                print_with_typing_effect(f"  - {ext}: {count} files", speed=0.01, color=Colors.YELLOW)
        else:
            # Get basic collection info from ChromaDB
            collection_info = rag_db.get_collection_info()
            print_with_typing_effect(f"Database Type: ChromaDB (local in-memory)", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Collection: {collection_info['name']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Document Count: {collection_info['count']}", speed=0.01, color=Colors.CYAN)
        
        return
    
    if args.sync_documents and config.USE_MILVUS:
        print_section_header("Document Synchronization")
        print_with_typing_effect("Synchronizing local documents with database...", speed=0.01)
        
        sync_result = rag_db.sync_documents(config.DOCUMENTS_PATH)
        if sync_result['status'] == 'success':
            print_with_typing_effect(f"Synchronization complete!", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect(f"New files: {sync_result['files_added']}", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect(f"Updated files: {sync_result['files_updated']}", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect(f"Unchanged files: {sync_result['files_unchanged']}", speed=0.01, color=Colors.CYAN)
            print_with_typing_effect(f"Deleted files: {sync_result['files_deleted']}", speed=0.01, color=Colors.YELLOW)
        else:
            print_with_typing_effect(f"Synchronization error: {sync_result['message']}", speed=0.01, color=Colors.RED)
        
        return
    
    if args.initial_index and config.USE_MILVUS:
        print_section_header("Initial Indexing")
        print_with_typing_effect("Performing initial indexing of all documents...", speed=0.01)
        
        # Process all documents in the directory
        documents = doc_processor.process_directory(config.DOCUMENTS_PATH)
        if documents:
            print_with_typing_effect(f"Processing {len(documents)} documents for indexing...", speed=0.01)
            
            # Index the documents
            success = rag_db.index_documents(documents)
            if success:
                print_with_typing_effect(f"Successfully indexed {len(documents)} documents.", speed=0.01, color=Colors.GREEN)
            else:
                print_with_typing_effect("Error indexing documents.", speed=0.01, color=Colors.RED)
        else:
            print_with_typing_effect("No documents found to index!", speed=0.01, color=Colors.RED)
        
        return
    
    if args.recreate_index and config.USE_MILVUS:
        print_section_header("Recreate Index")
        print_with_typing_effect("Recreating index from scratch...", speed=0.01)
        
        result = rag_db.recreate_index()
        if result['status'] == 'success':
            print_with_typing_effect(f"Index recreated successfully.", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect("You should run --initial-index to populate the new index.", speed=0.01, color=Colors.YELLOW)
        else:
            print_with_typing_effect(f"Error recreating index: {result['message']}", speed=0.01, color=Colors.RED)
        
        return
    
    if args.reindex_file and config.USE_MILVUS:
        print_section_header("Reindex File")
        print_with_typing_effect(f"Reindexing file: {args.reindex_file}...", speed=0.01)
        
        # Find files matching the pattern
        import glob
        file_path = Path(args.reindex_file)
        if not file_path.is_absolute():
            file_path = config.DOCUMENTS_PATH / args.reindex_file
        
        matching_files = glob.glob(str(file_path))
        if not matching_files:
            print_with_typing_effect(f"No files found matching pattern: {args.reindex_file}", speed=0.01, color=Colors.RED)
            return
        
        print_with_typing_effect(f"Found {len(matching_files)} files to reindex", speed=0.01)
        
        # Reindex each file
        success_count = 0
        for file in matching_files:
            file_path = Path(file)
            print_with_typing_effect(f"Reindexing {file_path.name}...", speed=0.01)
            success = rag_db._reindex_file(file_path)
            if success:
                success_count += 1
                print_with_typing_effect(f"Successfully reindexed {file_path.name}", speed=0.01, color=Colors.GREEN)
            else:
                print_with_typing_effect(f"Error reindexing {file_path.name}", speed=0.01, color=Colors.RED)
        
        print_with_typing_effect(f"Reindexed {success_count} out of {len(matching_files)} files", speed=0.01, color=Colors.CYAN)
        return
    
    if args.export_index and config.USE_MILVUS:
        print_section_header("Export Index")
        export_path = args.export_index
        print_with_typing_effect(f"Exporting index to {export_path}...", speed=0.01)
        
        result = rag_db.export_index(export_path)
        if result['status'] == 'success':
            print_with_typing_effect(f"Index exported successfully to {export_path}", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect(f"Exported {result['file_count']} files with {result['chunk_count']} chunks", speed=0.01, color=Colors.CYAN)
        else:
            print_with_typing_effect(f"Error exporting index: {result['message']}", speed=0.01, color=Colors.RED)
        
        return
    
    if args.import_index and config.USE_MILVUS:
        print_section_header("Import Index")
        import_path = args.import_index
        print_with_typing_effect(f"Importing index from {import_path}...", speed=0.01)
        
        result = rag_db.import_index(import_path)
        if result['status'] == 'success':
            print_with_typing_effect(f"Index imported successfully from {import_path}", speed=0.01, color=Colors.GREEN)
            print_with_typing_effect(f"Imported {result['file_count']} files with {result['chunk_count']} chunks", speed=0.01, color=Colors.CYAN)
        else:
            print_with_typing_effect(f"Error importing index: {result['message']}", speed=0.01, color=Colors.RED)
        
        return

    # Initialize travel agent with the database
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
    if args.chat_mode:
        # Chat mode - use chat-based plan generation
        config.USE_CHAT_MODE = True
        print_with_typing_effect("🌟 ADVANCED MODE ENABLED: Using multi-section plan generation",
                             speed=0.01, color=Colors.GREEN + Colors.BOLD)
        print_with_typing_effect("    - Generates each section independently for maximum detail", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - Produces vivid, inspiring travel descriptions", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - Creates comprehensive, detailed itineraries", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - NO placeholder references or missing sections", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - Includes rich metadata with percentage analysis", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - Advanced evaluation and recommendations", speed=0.01, color=Colors.GREEN)
        print_with_typing_effect("    - NEW: Plausibility check to verify plan logic and request matching", speed=0.01, color=Colors.YELLOW + Colors.BOLD)

    if args.force_mode:
        # Force mode - aggressively skip all rate limiting
        config.OFFLINE_MODE = True  # Keep the variable name for compatibility
        config.SKIP_METADATA_GENERATION = True
        config.OPTIMIZE_KEYWORD_EXTRACTION = True
        config.SKIP_DRAFT_PLAN = True
        print_with_typing_effect("⚡ FORCE MODE ENABLED: Maximum speed with NO rate limiting",
                             speed=0.01, color=Colors.RED + Colors.BOLD)
        print_with_typing_effect("    - CAUTION: May use API quota rapidly", speed=0.01, color=Colors.RED)
        print_with_typing_effect("    - All delays and rate limiting disabled", speed=0.01, color=Colors.RED)
        print_with_typing_effect("    - Fastest possible operation", speed=0.01, color=Colors.YELLOW)
        print_with_typing_effect("    - Use with care during heavy usage periods", speed=0.01, color=Colors.RED)
    elif args.minimal_mode:
        # Absolute minimal mode - just 2 API calls
        config.SKIP_METADATA_GENERATION = True
        config.OPTIMIZE_KEYWORD_EXTRACTION = False
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

        if hasattr(config, 'USE_CHAT_MODE') and config.USE_CHAT_MODE:
            print_with_typing_effect(f"  CHAT MODE: {'Enabled' if config.USE_CHAT_MODE else 'Disabled'}",
                                speed=0.01, color=Colors.GREEN + Colors.BOLD)

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
        print_with_typing_effect("  - Chat mode: 2 calls (1× keywords + 1 continuous chat session)",
                            speed=0.01, color=Colors.GREEN + Colors.BOLD)

        print()
        return

    # Force reindexing if requested
    force_reindex = args.reindex

    # Handle reindexing based on database type
    if config.USE_MILVUS:
        # For Milvus, we handle reindexing through CLI commands
        if force_reindex:
            print_section_header("Database Reindexing")
            print_with_typing_effect("Reindexing with Milvus/Zilliz...", speed=0.01)
            
            # First recreate the index
            result = rag_db.recreate_index()
            if result['status'] == 'success':
                print_with_typing_effect("Index recreated successfully.", speed=0.01, color=Colors.GREEN)
                
                # Now process documents and index them
                print_with_typing_effect("Processing documents for indexing...", speed=0.01)
                documents = doc_processor.process_directory(config.DOCUMENTS_PATH)
                if documents:
                    print_with_typing_effect(f"Indexing {len(documents)} document chunks...", speed=0.01)
                    success = rag_db.index_documents(documents)
                    if success:
                        print_with_typing_effect(f"Indexed {len(documents)} document chunks into Milvus.",
                                             speed=0.01, color=Colors.GREEN)
                    else:
                        print_with_typing_effect("Error indexing documents!", speed=0.01, color=Colors.RED)
                else:
                    print_with_typing_effect("No documents found to index!", speed=0.01, color=Colors.RED)
            else:
                print_with_typing_effect(f"Error recreating index: {result['message']}", speed=0.01, color=Colors.RED)
    else:
        # For ChromaDB, use the original reindexing approach
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

        # Check if ChromaDB needs to be initialized with documents
        if (force_reindex or not os.path.exists(config.CHROMA_DB_PATH) or 
                len(os.listdir(config.CHROMA_DB_PATH)) == 0):
            print_section_header("Database Initialization")
            print_with_typing_effect("Initializing ChromaDB with travel documents...", speed=0.02)
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
    
    if config.USE_MILVUS:
        print_with_typing_effect(f"Database Type: Milvus/Zilliz Cloud", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Server: {collection_info.get('server', 'Unknown')}", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Collection: {collection_info['name']}", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Documents: {collection_info['count']}", speed=0.01, color=Colors.CYAN)
        
        # Show last indexed time if available
        if 'last_indexed' in collection_info and collection_info['last_indexed']:
            print_with_typing_effect(f"Last indexed: {collection_info['last_indexed']}", 
                                 speed=0.01, color=Colors.CYAN)
            
        # Show indexed files count if available
        if 'indexed_files' in collection_info:
            print_with_typing_effect(f"Indexed files: {collection_info['indexed_files']}", 
                                 speed=0.01, color=Colors.CYAN)
    else:
        print_with_typing_effect(f"Database Type: ChromaDB (local in-memory)", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Collection: {collection_info['name']}", speed=0.01, color=Colors.CYAN)
        print_with_typing_effect(f"Documents: {collection_info['count']}", speed=0.01, color=Colors.CYAN)

    # Create output directory for saved plans
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    print_section_header("Welcome")
    print_with_typing_effect("I'm your travel planning assistant!", speed=0.02)
    print_with_typing_effect("Tell me about your dream vacation, and I'll create a personalized travel plan.", speed=0.02)
    print_with_typing_effect("You can type 'quit' or 'exit' at any time to end our session.", speed=0.02)

    # Check if prompt was provided from file or command line
    predefined_prompt = None

    if args.prompt:
        try:
            with open(args.prompt, 'r', encoding='utf-8') as f:
                predefined_prompt = f.read().strip()
                print_with_typing_effect(f"Using prompt from file: {args.prompt}", speed=0.01, color=Colors.GREEN)
        except Exception as e:
            print_with_typing_effect(f"Error reading prompt file: {e}", speed=0.01, color=Colors.RED)

    elif args.prompt_text:
        predefined_prompt = args.prompt_text
        print_with_typing_effect("Using prompt provided via command line", speed=0.01, color=Colors.GREEN)

    while True:
        # Determine if we're using a predefined prompt or user input
        if predefined_prompt:
            user_query = predefined_prompt
            print_with_typing_effect("Using predefined prompt for travel planning...", speed=0.01, color=Colors.CYAN)
            # Reset predefined_prompt so that after processing, the program exits
            prompt_to_process = predefined_prompt
            predefined_prompt = None
        else:
            # Interactive mode with normal user input
            user_query = ask_user("What kind of trip are you planning?")

            if user_query.lower() in ["quit", "exit"]:
                print_with_typing_effect("Thank you for using the Travel Plan Agent. Goodbye!",
                                     speed=0.02, color=Colors.CYAN)
                break

            prompt_to_process = user_query

        # Process user query and generate travel plan
        try:
            should_save_plan = True  # Flag to determine if we should save the plan
            detailed_plan = ""  # Initialize detailed_plan variable

            # Check if we should use the chat-based approach
            if config.USE_CHAT_MODE:
                print_section_header("Multi-section Plan Creation")
                print_with_typing_effect("Using advanced multi-section travel plan generation...",
                                    speed=0.02, color=Colors.GREEN)

                # Show search process in real-time
                print_section_header("RAG Search Process")
                print_with_typing_effect("Searching knowledge base for relevant information...", speed=0.01)

                # Generate the complete plan using a multi-section approach
                print_section_header("Creating Travel Plan")
                print_with_typing_effect("Analyzing your request for optimal planning...", speed=0.02, color=Colors.BLUE)
                print_with_typing_effect("Generating comprehensive travel plan in multiple detailed sections...", speed=0.02)
                print_with_typing_effect("Ensuring plan logic and verifying request match with plausibility check...",
                                   speed=0.01, color=Colors.YELLOW)
                print_with_typing_effect("This approach ensures complete, vivid descriptions tailored exactly to your request!",
                                   speed=0.01, color=Colors.GREEN)
                detailed_plan = agent.generate_travel_plan_chat(prompt_to_process)

                print_section_header("Travel Plan")
                print(Colors.CYAN + detailed_plan + Colors.END)

            # Otherwise, use the traditional approach
            else:
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
                    draft_plan = agent.generate_draft_plan(prompt_to_process)

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

                    # Generate plan and metadata separately
                    print_section_header("Creating Comprehensive Plan")
                    print_with_typing_effect("Step 1/2: Creating comprehensive travel plan...", speed=0.01)
                    detailed_plan, metadata_report = agent.generate_detailed_plan(prompt_to_process, draft_plan, feedback)

                    print_section_header("Detailed Travel Plan")
                    print(Colors.CYAN + detailed_plan + Colors.END)
                else:
                    # User doesn't want to proceed with this plan
                    should_save_plan = False
                    print_with_typing_effect("Operation cancelled. Let's try again!",
                                        speed=0.02, color=Colors.YELLOW)

            # Display token usage summary
            if hasattr(agent, 'token_tracker'):
                print_section_header("Token Usage Summary")
                token_summary = agent.token_tracker.generate_summary()
                token_chart = agent.token_tracker.generate_ascii_chart()
                
                # Use colors for the output
                print(Colors.CYAN + token_summary + token_chart + Colors.END)
                
                # Check if tokens were saved through context optimization
                if agent.token_tracker.tokens_saved > 0:
                    token_savings_pct = (agent.token_tracker.tokens_saved / agent.token_tracker.tokens_used['total_tokens']) * 100
                    print_with_typing_effect(f"Saved {agent.token_tracker.tokens_saved:,} tokens ({token_savings_pct:.1f}%) with context optimization!",
                                        speed=0.01, color=Colors.GREEN + Colors.BOLD)
            
            # Save the plan if we should
            if should_save_plan and detailed_plan:
                # Save the plan with its title - pass both plan and metadata to save_travel_plan
                print_with_typing_effect("Saving your travel plan...", speed=0.02)
                if 'metadata_report' in locals():
                    saved_files = save_travel_plan(detailed_plan, metadata_report)
                else:
                    # For chat mode or if metadata wasn't generated
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

        except Exception as e:
            logger.error(f"Error generating travel plan: {str(e)}", exc_info=True)
            print_with_typing_effect(f"An error occurred: {str(e)}",
                                 speed=0.02, color=Colors.RED)

if __name__ == "__main__":
    main()
