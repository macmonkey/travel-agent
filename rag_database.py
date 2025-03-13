"""
RAG database module for the Travel Plan Agent.
This module contains functionality for managing the vector database using ChromaDB.
"""

from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

class RAGDatabase:
    """Class for managing the RAG vector database with ChromaDB."""
    
    def __init__(self, config):
        """
        Initialize the RAG database.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.config.CHROMA_DB_PATH))
        
        # Set up default embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="travel_documents",
            embedding_function=self.embedding_function,
            metadata={"description": "Travel documents for personalized travel planning"}
        )
    
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
            doc_id = f"doc_{i}_{doc['metadata']['filename']}_{doc['metadata']['chunk_id']}"
            ids.append(doc_id)
            texts.append(doc['text'])
            metadatas.append(doc['metadata'])
        
        # Add documents to the collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print(f"Indexed {len(ids)} document chunks in the database.")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Process and format the results
        formatted_results = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                formatted_result = {
                    'text': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': results['distances'][0][i] if results['distances'] else None,
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Get relevant context as a concatenated string for a given query.
        
        Args:
            query: The search query
            n_results: Number of results to include
            
        Returns:
            Concatenated string of relevant document chunks
        """
        results = self.search(query, n_results)
        
        if not results:
            return "No relevant information found in the database."
        
        # Format results as a single context string
        context_parts = []
        for i, result in enumerate(results):
            source = result['metadata'].get('source', 'Unknown source')
            context_parts.append(f"DOCUMENT {i+1} (Source: {source}):\n{result['text']}\n")
        
        return "\n".join(context_parts)