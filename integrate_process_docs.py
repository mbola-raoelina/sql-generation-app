#!/usr/bin/env python3
"""
Integrate process documentation into ChromaDB while preserving existing embeddings.
This script adds business process context to the existing schema embeddings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chroma_db"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PROCESS_DOCS_DIR = "./process_docs"
COLLECTION_NAME = "schema_docs"

def load_process_documents() -> Dict[str, str]:
    """
    Load all process documentation files from the process_docs directory.
    Returns a dictionary mapping process names to their content.
    """
    process_docs = {}
    
    if not os.path.exists(PROCESS_DOCS_DIR):
        logger.warning(f"Process docs directory {PROCESS_DOCS_DIR} not found")
        return process_docs
    
    for file_path in Path(PROCESS_DOCS_DIR).glob("*.txt"):
        process_name = file_path.stem.upper()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    process_docs[process_name] = content
                    logger.info(f"Loaded process document: {process_name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return process_docs

def create_process_embeddings(process_docs: Dict[str, str], model) -> List[Dict[str, Any]]:
    """
    Create embeddings for process documentation.
    Returns a list of documents ready for ChromaDB insertion.
    """
    documents = []
    
    for process_name, content in process_docs.items():
        # Create multiple chunks for better retrieval
        chunks = create_content_chunks(content, process_name)
        
        for i, chunk in enumerate(chunks):
            doc_id = f"process_{process_name}_{i}"
            
            document = {
                "id": doc_id,
                "content": chunk,
                "metadata": {
                    "type": "process_documentation",
                    "process_name": process_name,
                    "chunk_index": i,
                    "source": f"process_docs/{process_name.lower()}.txt",
                    "document_type": "business_process"
                }
            }
            documents.append(document)
    
    return documents

def create_content_chunks(content: str, process_name: str) -> List[str]:
    """
    Split process documentation into meaningful chunks for better retrieval.
    """
    chunks = []
    
    # Split by sections (STEPS, KEY RELATIONSHIPS, COMMON QUERIES)
    sections = content.split('\n\n')
    current_chunk = ""
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # If section is too long, split it further
        if len(section) > 500:
            # Split by lines and group them
            lines = section.split('\n')
            temp_chunk = ""
            
            for line in lines:
                if len(temp_chunk + line) > 500:
                    if temp_chunk:
                        chunks.append(f"PROCESS: {process_name}\n\n{temp_chunk.strip()}")
                    temp_chunk = line + "\n"
                else:
                    temp_chunk += line + "\n"
            
            if temp_chunk:
                chunks.append(f"PROCESS: {process_name}\n\n{temp_chunk.strip()}")
        else:
            # Add to current chunk or create new one
            if len(current_chunk + section) > 500:
                if current_chunk:
                    chunks.append(f"PROCESS: {process_name}\n\n{current_chunk.strip()}")
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(f"PROCESS: {process_name}\n\n{current_chunk.strip()}")
    
    return chunks

def integrate_process_docs():
    """
    Main function to integrate process documentation into ChromaDB.
    """
    logger.info("Starting process documentation integration...")
    
    # Load process documents
    process_docs = load_process_documents()
    if not process_docs:
        logger.warning("No process documents found to integrate")
        return
    
    logger.info(f"Found {len(process_docs)} process documents")
    
    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Connected to existing ChromaDB collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
        return
    
    # Load embedding model
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info(f"Loaded embedding model: {EMBED_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return
    
    # Create process document embeddings
    process_documents = create_process_embeddings(process_docs, model)
    logger.info(f"Created {len(process_documents)} process document chunks")
    
    # Add to ChromaDB
    try:
        # Extract data for ChromaDB
        ids = [doc["id"] for doc in process_documents]
        contents = [doc["content"] for doc in process_documents]
        metadatas = [doc["metadata"] for doc in process_documents]
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(process_documents)} process documents to ChromaDB")
        
        # Verify integration
        count = collection.count()
        logger.info(f"Total documents in collection: {count}")
        
        # Test retrieval
        test_query = "procure to pay process"
        results = collection.query(
            query_texts=[test_query],
            n_results=3,
            where={"type": "process_documentation"}
        )
        
        if results['documents'] and results['documents'][0]:
            logger.info("✅ Process documentation integration successful!")
            logger.info(f"Test query '{test_query}' returned {len(results['documents'][0])} results")
        else:
            logger.warning("⚠️ Process documentation integration completed but test query returned no results")
            
    except Exception as e:
        logger.error(f"Error adding process documents to ChromaDB: {e}")
        return
    
    logger.info("Process documentation integration completed successfully!")

def create_process_context_enhancer():
    """
    Create a function that enhances SQL generation with process context.
    This will be integrated into the main sqlgen.py system.
    """
    enhancer_code = '''
def get_process_context_for_query(user_query: str, collection) -> str:
    """
    Retrieve relevant process documentation context for a user query.
    This enhances the LLM's understanding of business processes.
    """
    try:
        # Query for process documentation relevant to the user query
        results = collection.query(
            query_texts=[user_query],
            n_results=3,
            where={"type": "process_documentation"}
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
        
        # Combine relevant process context
        process_context = "\\n\\nBUSINESS PROCESS CONTEXT:\\n"
        for i, doc in enumerate(results['documents'][0]):
            process_context += f"\\n--- Process Context {i+1} ---\\n"
            process_context += doc + "\\n"
        
        return process_context
        
    except Exception as e:
        logger.warning(f"Error retrieving process context: {e}")
        return ""
'''
    
    return enhancer_code

if __name__ == "__main__":
    integrate_process_docs()
