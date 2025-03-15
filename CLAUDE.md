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

## Database Options
The travel agent supports two vector database backends:
1. **ChromaDB** (default) - Local in-memory vector database
2. **Milvus/Zilliz Cloud** - Remote cloud-based vector database with persistent storage

### Milvus/Zilliz Integration
- The agent can connect to a Zilliz Cloud Serverless instance for improved performance and persistence
- Uses API token authentication for secure connection
- Uses optimized chunking with sentence boundary detection
- Implements a two-phase retrieval for better results
- Supports document change detection and selective reindexing

## Common Commands

### Basic Usage
- Run the application: `python main.py`
- Process documents: `python document_processor.py`

### Milvus/Zilliz Commands
- Test Milvus connection: `python test_milvus.py --verbose`
- Run with Milvus: `python main.py --use-milvus`
- Show index status: `python main.py --use-milvus --index-status`
- Initial indexing: `python main.py --use-milvus --initial-index`
- Synchronize documents: `python main.py --use-milvus --sync-documents`
- Reindex specific file: `python main.py --use-milvus --reindex-file <filename>`
- Export index backup: `python main.py --use-milvus --export-index <backup_file.json>`
- Import index backup: `python main.py --use-milvus --import-index <backup_file.json>`
- Recreate index: `python main.py --use-milvus --recreate-index`