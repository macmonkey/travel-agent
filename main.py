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

def main():
    """Main function to run the Travel Plan Agent application."""
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
    
    # Check if database needs to be initialized with documents
    if not os.path.exists(config.CHROMA_DB_PATH) or len(os.listdir(config.CHROMA_DB_PATH)) == 0:
        print_section_header("Database Initialization")
        print_with_typing_effect("Initializing database with travel documents...", speed=0.02)
        documents = doc_processor.process_directory(config.DOCUMENTS_PATH)
        rag_db.index_documents(documents)
        print_with_typing_effect(f"Indexed {len(documents)} documents into the database.", 
                             speed=0.02, color=Colors.GREEN)
    
    # Create output directory for saved plans
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize travel agent with the RAG database
    agent = TravelAgent(config, rag_db)
    
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
            # Generate draft plan
            print_section_header("Creating Draft Plan")
            print_with_typing_effect("Analyzing your preferences and searching for relevant information...", speed=0.02)
            draft_plan = agent.generate_draft_plan(user_query)
            
            print_section_header("Draft Travel Plan")
            print(Colors.CYAN + draft_plan + Colors.END)
            
            # Ask for user feedback
            feedback = ask_user("Do you like this plan? Any changes or specific requests?")
            
            if feedback.lower() not in ["no", "quit", "exit"]:
                # Generate detailed plan with user feedback
                print_section_header("Creating Detailed Plan")
                print_with_typing_effect("Incorporating your feedback and creating a detailed itinerary...", speed=0.02)
                detailed_plan = agent.generate_detailed_plan(user_query, draft_plan, feedback)
                
                print_section_header("Detailed Travel Plan")
                print(Colors.CYAN + detailed_plan + Colors.END)
                
                # Extract destination name for file naming
                destination_match = re.search(r"# ([^#\n]+)", detailed_plan)
                destination_name = destination_match.group(1).strip() if destination_match else "Travel Plan"
                
                # Save the plan
                print_with_typing_effect("Saving your travel plan...", speed=0.02)
                saved_files = save_travel_plan(detailed_plan, destination_name)
                
                print_section_header("Plan Saved")
                print_with_typing_effect(f"Markdown: {saved_files['markdown']}", speed=0.01, color=Colors.GREEN)
                print_with_typing_effect(f"Text: {saved_files['txt']}", speed=0.01, color=Colors.GREEN)
                
                if saved_files.get('html'):
                    print_with_typing_effect(f"HTML: {saved_files['html']}", speed=0.01, color=Colors.GREEN)
                    
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