#!/usr/bin/env python3
"""
Script to inspect and debug ChromaDB database issues.
"""

import os
import sys
import time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Path to the ChromaDB database
CHROMA_DB_PATH = "./data/chromadb"

def inspect_database():
    """Inspect the ChromaDB database and print diagnostics."""
    print(f"Inspecting ChromaDB at: {CHROMA_DB_PATH}")
    
    # Check if the directory exists
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB directory does not exist at {CHROMA_DB_PATH}")
        return
    
    # Check if the SQLite file exists
    sqlite_file = os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")
    if os.path.exists(sqlite_file):
        # Get file size and modification time
        file_size = os.path.getsize(sqlite_file)
        mod_time = os.path.getmtime(sqlite_file)
        
        print(f"chroma.sqlite3 file size: {file_size} bytes")
        print(f"Last modified: {time.ctime(mod_time)}")
    else:
        print(f"Note: chroma.sqlite3 file doesn't exist yet - will be created on first use")
    
    try:
        # Initialize ChromaDB client
        print("Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # List all collections
        collection_names = client.list_collections()
        print(f"Number of collections: {len(collection_names)}")
        
        if collection_names:
            for name in collection_names:
                print(f"\nCollection: {name}")
                try:
                    coll = client.get_collection(name)
                    print(f"Metadata: {coll.metadata}")
                except Exception as e:
                    print(f"Error getting collection metadata: {e}")
                
                # Get document count
                try:
                    coll = client.get_collection(name)
                    count = coll.count()
                    print(f"Document count: {count}")
                    
                    # Try to get some sample documents
                    if count > 0:
                        try:
                            results = coll.peek(limit=3)
                            print(f"Sample documents: {results}")
                        except Exception as e:
                            print(f"Error peeking at documents: {e}")
                    else:
                        print("Collection is empty (0 documents)")
                except Exception as e:
                    print(f"Error accessing collection: {e}")
                    
                # Try to embed a simple text to verify embedding function works
                try:
                    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
                    sample_embedding = embedding_function(["This is a test document"])
                    print(f"Embedding function works. Output shape: {len(sample_embedding[0])}")
                except Exception as e:
                    print(f"Error testing embedding function: {e}")
        else:
            print("No collections found in the database")
        
        # Try to create and add to a test collection
        print("\nAttempting to create a test collection...")
        try:
            # Delete test collection if it exists
            try:
                client.delete_collection("test_collection")
                print("Deleted existing test collection")
            except:
                pass
            
            # Create test collection
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
            test_coll = client.create_collection(
                name="test_collection",
                embedding_function=embedding_function
            )
            
            # Add a test document with simple metadata
            try:
                test_coll.add(
                    documents=["This is a test document with simple metadata"],
                    metadatas=[{"source": "test", "filename": "test.txt"}],
                    ids=["test1"]
                )
                print("Successfully added document with simple metadata")
            except Exception as e:
                print(f"Error adding document with simple metadata: {e}")
            
            # Add a test document with complex metadata (similar to what the app tries to do)
            try:
                complex_metadata = {
                    "source": "test_complex",
                    "filename": "test_complex.txt",
                    "chunk_id": 1,
                    "entities": {
                        "locations": "Berlin; Paris; London",
                        "hotels": "Marriott Hotel; Hilton",
                        "activities": "City tour; Museum visit; Beach relaxation"
                    }
                }
                test_coll.add(
                    documents=["This is a test document with complex metadata"],
                    metadatas=[complex_metadata],
                    ids=["test2"]
                )
                print("Successfully added document with complex metadata")
            except Exception as e:
                print(f"Error adding document with complex metadata: {e}")
                print("This suggests the issue is with the metadata structure")
            
            # Verify they were added
            count = test_coll.count()
            print(f"Test collection document count: {count}")
            
            if count > 0:
                results = test_coll.peek()
                print(f"Test collection contents: {results}")
            
            # Try adding documents in a batch
            batch_ids = [f"batch_{i}" for i in range(5)]
            batch_docs = [f"This is batch document {i}" for i in range(5)]
            batch_metadata = [{"source": f"batch_{i}", "filename": f"batch_{i}.txt"} for i in range(5)]
            
            try:
                test_coll.add(
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print("Successfully added batch documents")
            except Exception as e:
                print(f"Error adding batch documents: {e}")
            
            # Verify all were added
            count = test_coll.count()
            print(f"Test collection document count after batch: {count}")
            
        except Exception as e:
            print(f"Error during test collection operations: {e}")
        
        # Clean up
        try:
            client.delete_collection("test_collection")
            print("Test collection deleted")
        except:
            pass
                
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")

def reset_and_create_test_docs():
    """Reset the database and add test travel documents."""
    print("Resetting database and adding test documents...")
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Delete existing travel_documents collection if it exists
        try:
            client.delete_collection("travel_documents")
            print("Deleted existing travel_documents collection")
        except:
            print("No existing travel_documents collection to delete")
        
        # Create a new collection
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        collection = client.create_collection(
            name="travel_documents",
            embedding_function=embedding_function
        )
        
        # Create some test travel documents with proper metadata
        documents = [
            "Vietnam is a beautiful country with amazing beaches and delicious food. Hanoi is the capital city with charming old quarter and street food.",
            "Cambodia is known for Angkor Wat, a UNESCO world heritage site. Siem Reap is a popular tourist destination.",
            "Thailand offers beautiful beaches in Phuket and cultural experiences in Bangkok. The Thai cuisine is world-renowned.",
            "Halong Bay in Vietnam features thousands of limestone islands and is perfect for cruises and kayaking adventures.",
            "German tourists often prefer cultural tours in Vietnam with visits to historical sites and authentic local experiences."
        ]
        
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        metadatas = [
            {
                "source": "test_vietnam.pdf",
                "filename": "test_vietnam.pdf",
                "chunk_id": 0,
                "entities_locations": "Vietnam; Hanoi",
                "entities_hotels": "Hanoi Hilton; Old Quarter Hotel",
                "entities_activities": "Street food tour; Old quarter walking"
            },
            {
                "source": "test_cambodia.pdf",
                "filename": "test_cambodia.pdf",
                "chunk_id": 0,
                "entities_locations": "Cambodia; Siem Reap; Angkor Wat",
                "entities_hotels": "Angkor Palace Resort",
                "entities_activities": "Temple tours; Angkor Wat sunrise"
            },
            {
                "source": "test_thailand.pdf",
                "filename": "test_thailand.pdf",
                "chunk_id": 0,
                "entities_locations": "Thailand; Bangkok; Phuket",
                "entities_hotels": "Bangkok Marriott; Phuket Beach Resort",
                "entities_activities": "Beach relaxation; City tours"
            },
            {
                "source": "test_halong.pdf",
                "filename": "test_halong.pdf",
                "chunk_id": 0,
                "entities_locations": "Vietnam; Halong Bay",
                "entities_hotels": "Halong Bay Cruise Ship",
                "entities_activities": "Cruising; Kayaking; Cave exploration"
            },
            {
                "source": "test_german.pdf",
                "filename": "test_german.pdf",
                "chunk_id": 0,
                "entities_locations": "Vietnam; Hanoi; Hue; Hoi An",
                "entities_hotels": "Heritage Hotel; Cultural Resort",
                "entities_activities": "Cultural tours; Historical sites; Local experiences"
            }
        ]
        
        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Verify they were added
        count = collection.count()
        print(f"Added {count} test documents to travel_documents collection")
        
        if count > 0:
            results = collection.peek()
            print(f"Sample document: {results['documents'][0]}")
            print(f"Sample metadata: {results['metadatas'][0]}")
        
        # Test a simple query
        query_results = collection.query(
            query_texts=["Vietnam beach vacation"],
            n_results=2
        )
        
        print(f"\nQuery results for 'Vietnam beach vacation':")
        for i, doc in enumerate(query_results['documents'][0]):
            print(f"Result {i+1}: {doc[:100]}...")
            print(f"Metadata: {query_results['metadatas'][0][i]}")
            print(f"Distance: {query_results['distances'][0][i]}")
            print()
        
    except Exception as e:
        print(f"Error during test document creation: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_and_create_test_docs()
    else:
        inspect_database()
        print("\nTip: Run with --reset flag to reset the database and add test documents")