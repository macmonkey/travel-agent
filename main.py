#!/usr/bin/env python3
"""
Main entry point for the Travel Plan Agent application.
This script initializes the agent and handles user interaction.
"""

import os
import re
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from rag_database import RAGDatabase
from agent import TravelAgent
from utils import save_travel_plan

def main():
    """Main function to run the Travel Plan Agent application."""
    # Initialize configuration
    config = Config()
    
    # Initialize document processor
    doc_processor = DocumentProcessor(config)
    
    # Initialize RAG database
    rag_db = RAGDatabase(config)
    
    # Check if database needs to be initialized with documents
    if not os.path.exists(config.CHROMA_DB_PATH) or len(os.listdir(config.CHROMA_DB_PATH)) == 0:
        print("Initializing database with travel documents...")
        documents = doc_processor.process_directory(config.DOCUMENTS_PATH)
        rag_db.index_documents(documents)
        print(f"Indexed {len(documents)} documents into the database.")
    
    # Create output directory for saved plans
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize travel agent with the RAG database
    agent = TravelAgent(config, rag_db)
    
    print("Welcome to the Travel Plan Agent!")
    print("Describe your ideal trip and I'll create a personalized travel plan for you.")
    print("Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_query = input("\nWhat kind of trip are you planning? ")
        
        if user_query.lower() in ["quit", "exit"]:
            print("Thank you for using the Travel Plan Agent. Goodbye!")
            break
            
        # Process user query and generate travel plan
        try:
            # Generate draft plan
            draft_plan = agent.generate_draft_plan(user_query)
            
            print("\n--- Draft Travel Plan ---")
            print(draft_plan)
            
            # Ask for user feedback
            feedback = input("\nDo you like this plan? Any changes or specific requests? ")
            
            if feedback.lower() not in ["no", "quit", "exit"]:
                # Generate detailed plan with user feedback
                detailed_plan = agent.generate_detailed_plan(user_query, draft_plan, feedback)
                
                print("\n--- Detailed Travel Plan ---")
                print(detailed_plan)
                
                # Extract destination name for file naming
                destination_match = re.search(r"# ([^#\n]+)", detailed_plan)
                destination_name = destination_match.group(1).strip() if destination_match else "Travel Plan"
                
                # Save the plan
                saved_files = save_travel_plan(detailed_plan, destination_name)
                
                print("\nYour travel plan has been saved:")
                print(f"- Markdown: {saved_files['markdown']}")
                if saved_files['pdf']:
                    print(f"- PDF: {saved_files['pdf']}")
                else:
                    print("- PDF export not available. Install markdown and weasyprint packages for PDF support.")
            else:
                print("Operation cancelled. Let's try again!")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()