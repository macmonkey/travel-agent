#!/usr/bin/env python3
"""
Test script for Milvus/Zilliz Cloud connection.
This script checks if the Milvus connection can be established using the provided credentials.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_milvus_connection(uri=None, token=None, verbose=False):
    """
    Test connection to Milvus/Zilliz Cloud.
    
    Args:
        uri: Milvus/Zilliz URI (if None, use environment variable)
        token: Milvus/Zilliz API token (if None, use environment variable)
        verbose: Whether to print detailed info
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Import pymilvus
        try:
            from pymilvus import connections
        except ImportError:
            print("Error: pymilvus package not installed.")
            print("Please install it using: pip install pymilvus>=2.3.0")
            return False
        
        # Get connection parameters
        milvus_uri = uri or os.environ.get("MILVUS_URI")
        milvus_token = token or os.environ.get("MILVUS_TOKEN")
        
        if not milvus_uri:
            print("Error: Milvus URI not provided.")
            print("Please set the MILVUS_URI environment variable or provide it as an argument.")
            return False
        
        if not milvus_token:
            print("Warning: Milvus API token not provided.")
            print("For Zilliz Serverless, an API token is required.")
        
        if verbose:
            print(f"Connecting to Milvus/Zilliz at: {milvus_uri}")
            print(f"Using token authentication: {'Yes' if milvus_token else 'No (anonymous)'}")
        
        # Connection parameters
        conn_params = {
            "uri": milvus_uri,
            "db_name": "default"
        }
        
        # Add authentication if provided
        if milvus_token:
            conn_params["token"] = milvus_token
        
        # Try to connect
        connections.connect(alias="default", **conn_params)
        
        # Check connection status
        if connections.has_connection("default"):
            print("✅ Successfully connected to Milvus/Zilliz!")
            
            # Check if we can list collections
            if verbose:
                try:
                    from pymilvus import utility
                    collections = utility.list_collections()
                    print(f"Available collections: {collections}")
                except Exception as e:
                    print(f"Warning: Could not list collections: {e}")
            
            return True
        else:
            print("❌ Failed to connect to Milvus/Zilliz.")
            return False
    
    except Exception as e:
        print(f"❌ Error connecting to Milvus/Zilliz: {e}")
        return False
    finally:
        try:
            # Disconnect if connected
            from pymilvus import connections
            if connections.has_connection("default"):
                connections.disconnect("default")
                if verbose:
                    print("Disconnected from Milvus/Zilliz.")
        except:
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Test Milvus/Zilliz Cloud connection")
    parser.add_argument("--uri", type=str, help="Milvus/Zilliz URI")
    parser.add_argument("--token", type=str, help="Milvus/Zilliz API token")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--timeout", type=int, default=10, help="Connection timeout in seconds")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Milvus/Zilliz Connection Test")
    print("============================")
    
    if args.timeout:
        # Set connection timeout
        os.environ["PYMILVUS_CONNECTION_TIMEOUT"] = str(args.timeout)
        print(f"Using connection timeout: {args.timeout} seconds")
    
    success = test_milvus_connection(
        uri=args.uri,
        token=args.token,
        verbose=args.verbose
    )
    
    if success:
        print("\nConnection test passed! ✅")
        print("You can use the --use-milvus option with main.py")
        sys.exit(0)
    else:
        print("\nConnection test failed! ❌")
        print("Please check your Milvus/Zilliz credentials and try again.")
        sys.exit(1)