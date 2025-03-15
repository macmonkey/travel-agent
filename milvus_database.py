"""
MilvusDB module for the Travel Plan Agent.
This module contains functionality for managing the vector database using PyMilvus.
"""

import re
import time
import json
import os
import hashlib
import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

from sentence_transformers import SentenceTransformer

class MilvusDatabase:
    """Class for managing the RAG vector database with PyMilvus."""
    
    def __init__(self, config, connection_args=None, force_recreate=False):
        """
        Initialize the Milvus database.
        
        Args:
            config: Application configuration object
            connection_args: Dictionary with Milvus connection parameters
                (defaults to Zilliz Free Tier if None)
            force_recreate: Whether to force recreate the collection
        """
        self.config = config
        self.collection_name = "travel_documents"
        
        # Index metadata storage
        self.index_metadata_file = Path(self.config.DATA_DIR) / "index_metadata.json"
        self.index_metadata = self._load_index_metadata()
        
        # Embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Connect to Milvus/Zilliz
        self._connect_to_milvus(connection_args)
        
        # Create or get collection
        self._init_collection(force_recreate)
        
        # Initialize document hash cache for tracking changes
        self.document_hashes = self._load_document_hashes()

    def _connect_to_milvus(self, connection_args=None):
        """
        Connect to Milvus/Zilliz server using provided parameters or defaults.
        
        Args:
            connection_args: Dictionary with connection parameters
        """
        # Default Zilliz Cloud connection parameters
        if connection_args is None:
            connection_args = {
                "uri": self.config.MILVUS_URI,
                "token": self.config.MILVUS_TOKEN,
                "db_name": "default"
            }
        
        # Connect to Milvus/Zilliz
        try:
            print(f"Connecting to Milvus/Zilliz at {connection_args.get('uri', 'default URI')}")
            connections.connect(
                alias="default",
                **connection_args
            )
            print("Successfully connected to Milvus/Zilliz")
        except Exception as e:
            print(f"Error connecting to Milvus/Zilliz: {e}")
            raise
    
    def _init_collection(self, force_recreate=False):
        """
        Initialize the collection - create if it doesn't exist or force_recreate is True.
        
        Args:
            force_recreate: Whether to force recreate the collection
        """
        # Check if collection exists and delete if force_recreate
        if utility.has_collection(self.collection_name):
            if force_recreate:
                utility.drop_collection(self.collection_name)
                print(f"Dropped existing collection: {self.collection_name}")
            else:
                self.collection = Collection(self.collection_name)
                print(f"Using existing collection: {self.collection_name}")
                return
        
        # Define collection schema
        id_field = FieldSchema(
            name="id", 
            dtype=DataType.VARCHAR,
            is_primary=True, 
            max_length=100
        )
        text_field = FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535
        )
        vector_field = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self.vector_dim
        )
        
        # Define schema for metadata fields (flattened for Milvus compatibility)
        metadata_fields = [
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="entities_locations", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="entities_hotels", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="entities_activities", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="has_must_see", dtype=DataType.VARCHAR, max_length=5),
            FieldSchema(name="importance_level", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="location_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="chunk_locations", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_themes", dtype=DataType.VARCHAR, max_length=500)
        ]
        
        # Create schema with all fields
        schema = CollectionSchema(
            fields=[id_field, text_field, vector_field] + metadata_fields,
            description="Travel documents for personalized travel planning"
        )
        
        # Create collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )
        print(f"Created new collection: {self.collection_name}")
        
        # Create index on vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
        
        print(f"Creating index on collection {self.collection_name}...")
        self.collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        print("Index created successfully")
        
        # Load collection to memory for search operations
        self.collection.load()
        print("Collection loaded into memory")
    
    def _load_index_metadata(self) -> Dict[str, Any]:
        """
        Load index metadata from file.
        
        Returns:
            Dictionary with index metadata
        """
        if not self.index_metadata_file.exists():
            return {
                "last_indexed": None,
                "document_count": 0,
                "indexed_files": {},
                "indexed_chunks": 0
            }
        
        try:
            with open(self.index_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading index metadata: {e}")
            return {
                "last_indexed": None,
                "document_count": 0,
                "indexed_files": {},
                "indexed_chunks": 0
            }
    
    def _save_index_metadata(self):
        """Save index metadata to file."""
        try:
            with open(self.index_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.index_metadata, f, indent=2)
            print(f"Index metadata saved to {self.index_metadata_file}")
        except Exception as e:
            print(f"Error saving index metadata: {e}")
    
    def _load_document_hashes(self) -> Dict[str, str]:
        """
        Load document hashes from the index metadata.
        
        Returns:
            Dictionary mapping filenames to hash values
        """
        if not self.index_metadata or "indexed_files" not in self.index_metadata:
            return {}
        
        return {
            filename: info["hash"] 
            for filename, info in self.index_metadata.get("indexed_files", {}).items()
        }
    
    def _calculate_document_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash for a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            MD5 hash string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            if not utility.has_collection(self.collection_name):
                return {"name": "none", "count": 0, "metadata": {}}
            
            self.collection = Collection(self.collection_name)
            
            # Get the count of documents in the collection
            count = self.collection.num_entities
            
            # Load index metadata for more information
            index_stats = self._load_index_metadata()
            
            # Get other collection information
            info = {
                "name": self.collection_name,
                "count": count,
                "last_indexed": index_stats.get("last_indexed", "Unknown"),
                "indexed_files": len(index_stats.get("indexed_files", {})),
                "indexed_chunks": index_stats.get("indexed_chunks", 0),
                "database_type": "Milvus/Zilliz",
                "server": self.config.MILVUS_URI if hasattr(self.config, "MILVUS_URI") else "Unknown",
            }
            
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"name": "unknown", "count": 0, "metadata": {}}
    
    def sync_documents(self, document_dir: Path) -> Dict[str, Any]:
        """
        Synchronize the database with local documents.
        
        Args:
            document_dir: Directory containing the documents
            
        Returns:
            Dictionary with synchronization results
        """
        if not document_dir.exists():
            return {
                "status": "error",
                "message": f"Document directory {document_dir} does not exist",
                "files_added": 0,
                "files_updated": 0,
                "files_unchanged": 0
            }
        
        # Get all document files
        file_extensions = [".pdf", ".docx", ".doc", ".md", ".txt"]
        all_files = []
        for ext in file_extensions:
            all_files.extend(list(document_dir.glob(f"**/*{ext}")))
        
        if not all_files:
            return {
                "status": "info",
                "message": f"No documents found in {document_dir}",
                "files_added": 0,
                "files_updated": 0,
                "files_unchanged": 0
            }
        
        # Track changes
        files_added = 0
        files_updated = 0
        files_unchanged = 0
        
        # Check each file against the hash database
        for file_path in all_files:
            rel_path = file_path.relative_to(document_dir)
            file_hash = self._calculate_document_hash(file_path)
            
            # Check if file exists in index and if hash is different
            if str(rel_path) in self.document_hashes:
                if self.document_hashes[str(rel_path)] != file_hash:
                    # File has changed, need to reindex
                    print(f"File changed: {rel_path}")
                    self._reindex_file(file_path)
                    files_updated += 1
                else:
                    # File unchanged
                    files_unchanged += 1
            else:
                # New file, need to index
                print(f"New file: {rel_path}")
                self._index_file(file_path)
                files_added += 1
        
        # Check for deleted files
        db_files = set(self.document_hashes.keys())
        current_files = {str(f.relative_to(document_dir)) for f in all_files}
        deleted_files = db_files - current_files
        
        for deleted_file in deleted_files:
            print(f"File removed: {deleted_file}")
            self._remove_file_from_index(deleted_file)
        
        # Update index metadata
        self.index_metadata["last_indexed"] = datetime.datetime.now().isoformat()
        self._save_index_metadata()
        
        return {
            "status": "success",
            "message": "Synchronization complete",
            "files_added": files_added,
            "files_updated": files_updated,
            "files_unchanged": files_unchanged,
            "files_deleted": len(deleted_files)
        }
    
    def _index_file(self, file_path: Path) -> bool:
        """
        Index a single file.
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            True if successful, False otherwise
        """
        from document_processor import DocumentProcessor
        
        try:
            # Create a document processor to handle the file
            doc_processor = DocumentProcessor(self.config)
            
            # Get document hash
            doc_hash = self._calculate_document_hash(file_path)
            
            # Process the file based on its type
            if file_path.suffix.lower() == '.pdf':
                chunks = doc_processor.process_pdf(file_path)
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                chunks = doc_processor.process_docx(file_path)
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                must_see_content = doc_processor._extract_must_see_section(text)
                chunks = doc_processor._split_and_create_documents(text, file_path, must_see_content)
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                must_see_content = doc_processor._extract_must_see_section(text)
                chunks = doc_processor._split_and_create_documents(text, file_path, must_see_content)
            else:
                print(f"Unsupported file type: {file_path}")
                return False
            
            # Index the chunks
            if chunks:
                self._add_chunks_to_index(chunks)
                
                # Update index metadata
                rel_path = str(file_path.relative_to(self.config.DOCUMENTS_PATH))
                self.index_metadata["indexed_files"][rel_path] = {
                    "hash": doc_hash,
                    "indexed_at": datetime.datetime.now().isoformat(),
                    "chunk_count": len(chunks)
                }
                self.document_hashes[rel_path] = doc_hash
                
                return True
            else:
                print(f"No chunks extracted from {file_path}")
                return False
        except Exception as e:
            print(f"Error indexing file {file_path}: {e}")
            return False
    
    def _reindex_file(self, file_path: Path) -> bool:
        """
        Reindex a file by removing its previous entries and indexing it again.
        
        Args:
            file_path: Path to the file to reindex
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove existing entries for this file
            rel_path = str(file_path.relative_to(self.config.DOCUMENTS_PATH))
            self._remove_file_from_index(rel_path)
            
            # Index the file again
            return self._index_file(file_path)
        except Exception as e:
            print(f"Error reindexing file {file_path}: {e}")
            return False
    
    def _remove_file_from_index(self, file_path: str) -> bool:
        """
        Remove a file from the index.
        
        Args:
            file_path: Relative path of the file to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Search for documents with this source
            query = f'filename == "{os.path.basename(file_path)}"'
            
            # Execute the query
            self.collection.load()
            search_result = self.collection.query(
                expr=query,
                output_fields=["id"]
            )
            
            if search_result:
                # Delete the documents
                ids = [item["id"] for item in search_result]
                id_list = ",".join([f'"{id}"' for id in ids])
                self.collection.delete(f'id in [{id_list}]')
                print(f"Removed {len(ids)} chunks for file {file_path}")
                
                # Update index metadata
                if file_path in self.index_metadata["indexed_files"]:
                    chunk_count = self.index_metadata["indexed_files"][file_path].get("chunk_count", 0)
                    self.index_metadata["indexed_chunks"] -= chunk_count
                    del self.index_metadata["indexed_files"][file_path]
                    
                if file_path in self.document_hashes:
                    del self.document_hashes[file_path]
                
                return True
            else:
                print(f"No entries found for file {file_path}")
                
                # Clean up metadata even if no entries found
                if file_path in self.index_metadata["indexed_files"]:
                    del self.index_metadata["indexed_files"][file_path]
                
                if file_path in self.document_hashes:
                    del self.document_hashes[file_path]
                
                return True
        except Exception as e:
            print(f"Error removing file {file_path} from index: {e}")
            return False
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents in the vector database.
        
        Args:
            documents: List of document chunks with metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            print("No documents to index.")
            return False
        
        return self._add_chunks_to_index(documents)
    
    def _add_chunks_to_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the index.
        
        Args:
            documents: List of document chunks with metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return False
        
        try:
            # Prepare documents for Milvus
            ids = []
            texts = []
            vectors = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Ensure we have valid text content
                if not doc.get('text'):
                    print(f"Skipping document {i} - no text content")
                    continue
                
                # Create a unique ID (ensure it's under 100 chars)
                filename = doc['metadata']['filename']
                chunk_id = doc['metadata']['chunk_id']
                
                # Truncate filename if needed to keep ID under 100 chars
                max_filename_len = 80  # Safe limit to ensure total ID stays under 100
                if len(filename) > max_filename_len:
                    # Use hash of filename to ensure uniqueness
                    import hashlib
                    filename_hash = hashlib.md5(filename.encode()).hexdigest()[:16]
                    filename = f"{filename[:60]}_{filename_hash}"
                
                doc_id = f"doc_{i}_{filename}_{chunk_id}"
                
                # Final safety check - ensure ID is under 100 chars
                if len(doc_id) > 95:  # Extra safety margin
                    doc_id = f"d_{i}_{hashlib.md5(doc_id.encode()).hexdigest()}"
                
                # Diagnostic info about the document being indexed
                print(f"Indexing document: {doc_id}")
                print(f"Text length: {len(doc['text'])} characters")
                print(f"First 100 chars: {doc['text'][:100]}...")
                
                # Generate embedding
                embedding = self.embedding_model.encode(doc['text'])
                
                ids.append(doc_id)
                texts.append(doc['text'])
                vectors.append(embedding.tolist())
                
                # Prepare metadata - ensure all required fields exist
                metadata = doc['metadata'].copy()
                metadatas.append(metadata)
            
            # Add documents to the collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                try:
                    # Record timestamp before adding batch
                    start_time = time.time()
                    
                    # Prepare batch data
                    batch_ids = ids[i:batch_end]
                    batch_texts = texts[i:batch_end]
                    batch_vectors = vectors[i:batch_end]
                    batch_metadatas = metadatas[i:batch_end]
                    
                    # Prepare batch data for Milvus insert
                    insert_data = [
                        batch_ids,  # id field
                        batch_texts,  # text field
                        batch_vectors,  # vector field
                    ]
                    
                    # Add metadata fields
                    for field_name in [
                        "source", "filename", "chunk_id", "entities_locations",
                        "entities_hotels", "entities_activities", "has_must_see",
                        "importance_level", "location_name", "chunk_locations",
                        "chunk_themes"
                    ]:
                        # Get the metadata values, using appropriate defaults if missing
                        if field_name == "chunk_id":
                            # Convert chunk_id to integer for INT64 field
                            field_values = [int(m.get(field_name, 0)) for m in batch_metadatas]
                        else:
                            # Use empty string as default for string fields and ensure they don't exceed max length
                            field_values = []
                            for m in batch_metadatas:
                                value = str(m.get(field_name, ""))
                                # Ensure value doesn't exceed max length of 500 chars
                                if len(value) > 490:  # Leave some buffer
                                    print(f"Warning: Truncating {field_name} that exceeds max length (was {len(value)} chars)")
                                    value = value[:490] + "..."
                                field_values.append(value)
                        
                        insert_data.append(field_values)
                    
                    # Insert into collection
                    self.collection.insert(insert_data)
                    
                    # Calculate time taken
                    elapsed = time.time() - start_time
                    print(f"Batch {i//batch_size + 1} indexed in {elapsed:.2f} seconds ({batch_end - i} documents)")
                    
                    # Update metadata
                    self.index_metadata["indexed_chunks"] += (batch_end - i)
                    
                except Exception as e:
                    print(f"Error indexing batch {i//batch_size + 1}: {e}")
                    return False
            
            # Verify documents were added
            self.collection.flush()
            count = self.collection.num_entities
            print(f"Collection now contains {count} document chunks")
            
            # Update index metadata
            self.index_metadata["document_count"] = count
            self.index_metadata["last_indexed"] = datetime.datetime.now().isoformat()
            self._save_index_metadata()
            
            return True
        
        except Exception as e:
            print(f"Error in _add_chunks_to_index: {e}")
            return False
    
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
        
        # Generate query embedding
        print(f"Searching with query: '{emphasized_query}'")
        query_embedding = self.embedding_model.encode(emphasized_query)
        
        # Load collection before search
        self.collection.load()
        
        # Search with vector similarity
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="vector",
            param=search_params,
            limit=n_results,
            output_fields=[
                "text", "source", "filename", "chunk_id", "entities_locations",
                "entities_hotels", "entities_activities", "has_must_see",
                "importance_level", "location_name", "chunk_locations", "chunk_themes"
            ]
        )
        
        # Process and format the results
        formatted_results = []
        must_see_results = []  # Special container for must-see items
        important_results = []  # Medium priority items
        
        if results and results[0]:
            for hit in results[0]:
                # Extract all metadata fields from hit object safely
                # For pymilvus 2.x, entities are accessed as attributes
                metadata = {}
                
                # Helper function to safely get attribute
                def safe_get_attr(obj, attr_name, default_value=""):
                    try:
                        return getattr(obj, attr_name, default_value)
                    except (AttributeError, TypeError):
                        try:
                            # As fallback, try dictionary-style access
                            return obj.get(attr_name, default_value)
                        except (AttributeError, TypeError):
                            return default_value
                
                # Extract all metadata attributes safely
                metadata["source"] = safe_get_attr(hit.entity, "source")
                metadata["filename"] = safe_get_attr(hit.entity, "filename")
                metadata["chunk_id"] = safe_get_attr(hit.entity, "chunk_id", 0)
                metadata["entities_locations"] = safe_get_attr(hit.entity, "entities_locations")
                metadata["entities_hotels"] = safe_get_attr(hit.entity, "entities_hotels")
                metadata["entities_activities"] = safe_get_attr(hit.entity, "entities_activities")
                metadata["has_must_see"] = safe_get_attr(hit.entity, "has_must_see", "no")
                metadata["importance_level"] = safe_get_attr(hit.entity, "importance_level")
                metadata["location_name"] = safe_get_attr(hit.entity, "location_name")
                metadata["chunk_locations"] = safe_get_attr(hit.entity, "chunk_locations")
                metadata["chunk_themes"] = safe_get_attr(hit.entity, "chunk_themes")
                
                # Create formatted result
                formatted_result = {
                    'text': safe_get_attr(hit.entity, "text"),
                    'metadata': metadata,
                    'score': getattr(hit, "distance", 0.0),  # Use hit.distance if available, or 0.0 as fallback
                }
                
                # Check if document contains location names mentioned in the query
                location_match = False
                for location in locations:
                    if location.lower() in formatted_result['text'].lower():
                        location_match = True
                        break
                
                # Apply tiered prioritization logic
                if metadata["has_must_see"] == "yes":
                    # Top priority: MUST SEE content
                    must_see_results.append(formatted_result)
                    if metadata["location_name"]:
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
        
        # Look for country-related documents directly using the search method
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
    
    def get_index_status(self) -> Dict[str, Any]:
        """
        Get detailed status of the index.
        
        Returns:
            Dictionary with index status information
        """
        # Load the index metadata
        index_stats = self._load_index_metadata()
        
        # Get collection information
        collection_info = self.get_collection_info()
        
        # Calculate statistics
        if "indexed_files" in index_stats:
            file_count = len(index_stats["indexed_files"])
            last_indexed = index_stats.get("last_indexed", "Never")
            
            # Calculate file types
            file_types = {}
            for filename in index_stats["indexed_files"].keys():
                ext = os.path.splitext(filename)[1].lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            # Calculate total chunks
            total_chunks = 0
            for file_info in index_stats["indexed_files"].values():
                total_chunks += file_info.get("chunk_count", 0)
            
            # Calculate average chunks per file
            avg_chunks = total_chunks / file_count if file_count > 0 else 0
        else:
            file_count = 0
            last_indexed = "Never"
            file_types = {}
            total_chunks = 0
            avg_chunks = 0
        
        # Build status report
        status = {
            "collection_name": collection_info["name"],
            "database_type": "Milvus/Zilliz",
            "server": self.config.MILVUS_URI if hasattr(self.config, "MILVUS_URI") else "Unknown",
            "last_indexed": last_indexed,
            "total_files": file_count,
            "total_chunks": total_chunks,
            "avg_chunks_per_file": avg_chunks,
            "file_types": file_types,
            "entity_count": collection_info["count"],
        }
        
        return status
    
    def export_index(self, export_path: str) -> Dict[str, Any]:
        """
        Export the index metadata and configuration for backup.
        
        Args:
            export_path: Path to export the index
            
        Returns:
            Dictionary with export results
        """
        try:
            export_data = {
                "metadata": self._load_index_metadata(),
                "config": {
                    "collection_name": self.collection_name,
                    "vector_dim": self.vector_dim,
                }
            }
            
            # Save to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Index exported to {export_path}",
                "file_count": len(export_data["metadata"].get("indexed_files", {})),
                "chunk_count": export_data["metadata"].get("indexed_chunks", 0)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error exporting index: {e}"
            }
    
    def import_index(self, import_path: str) -> Dict[str, Any]:
        """
        Import the index metadata and configuration from backup.
        
        Args:
            import_path: Path to import the index from
            
        Returns:
            Dictionary with import results
        """
        try:
            # Load import data
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate import data
            if "metadata" not in import_data or "config" not in import_data:
                return {
                    "status": "error",
                    "message": "Invalid import file format"
                }
            
            # Check compatibility
            if import_data["config"].get("vector_dim") != self.vector_dim:
                return {
                    "status": "error",
                    "message": f"Vector dimensions do not match: import has {import_data['config'].get('vector_dim')}, current is {self.vector_dim}"
                }
            
            # Update metadata
            self.index_metadata = import_data["metadata"]
            self._save_index_metadata()
            
            # Reload document hashes
            self.document_hashes = self._load_document_hashes()
            
            return {
                "status": "success",
                "message": f"Index imported from {import_path}",
                "file_count": len(import_data["metadata"].get("indexed_files", {})),
                "chunk_count": import_data["metadata"].get("indexed_chunks", 0)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error importing index: {e}"
            }
    
    def recreate_index(self) -> Dict[str, Any]:
        """
        Completely recreate the index from scratch.
        
        Returns:
            Dictionary with recreation results
        """
        try:
            # First drop the existing collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"Dropped existing collection: {self.collection_name}")
            
            # Reset index metadata
            self.index_metadata = {
                "last_indexed": None,
                "document_count": 0,
                "indexed_files": {},
                "indexed_chunks": 0
            }
            self._save_index_metadata()
            
            # Clear document hashes
            self.document_hashes = {}
            
            # Recreate the collection
            self._init_collection(force_recreate=True)
            
            return {
                "status": "success",
                "message": "Index recreated successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error recreating index: {e}"
            }