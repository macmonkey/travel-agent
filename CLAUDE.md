# Travel Agent Project Information

## Project Overview
This project is a travel planning agent built with Google Gemini API. It uses a RAG (Retrieval-Augmented Generation) approach to generate personalized travel plans based on user queries and a knowledge base of travel-related content.

## Important Notes

### Output Formats
- The travel agent should only output plans in **Markdown (.md)** and **Plain Text (.txt)** formats
- **DO NOT** implement HTML export functionality - this has been explicitly requested by the user
- Previous attempts to add HTML output have been removed

### Key Features
- Semantic analysis of user queries
- Multi-stage travel plan generation
- Enhanced keyword extraction
- Tiered search prioritization for "MUST SEE" content
- Improved document processing with multiple priority levels
- Optimized filename generation with comprehensive metadata

## Common Commands
- Run the application: `python main.py`
- Process documents: `python document_processor.py`