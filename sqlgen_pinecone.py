"""
Pinecone integration for sqlgen.py - replaces local ChromaDB for Streamlit Cloud deployment.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import pinecone
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Global caches
_pinecone_index_cache = None
_embed_model_cache = None

def get_pinecone_index():
    """Get cached Pinecone index or create it once"""
    global _pinecone_index_cache
    
    if _pinecone_index_cache is None:
        try:
            # Load configuration from environment variables
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            
            if not all([api_key, environment, index_name]):
                raise ValueError("Missing Pinecone environment variables: PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME")
            
            # Initialize Pinecone client
            pc = pinecone.Pinecone(api_key=api_key)
            
            # Get index
            _pinecone_index_cache = pc.Index(index_name)
            logger.info("Pinecone index connected and cached")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise e
    
    return _pinecone_index_cache

def get_embed_model():
    """Get cached embedding model or load it once"""
    global _embed_model_cache
    
    if _embed_model_cache is None:
        try:
            logger.info("Loading embedding model: thenlper/gte-base")
            _embed_model_cache = SentenceTransformer(
                'thenlper/gte-base',
                device='cpu',
                trust_remote_code=True
            )
            logger.info("Embedding model loaded and cached")
        except Exception as e:
            logger.warning(f"Failed to load thenlper/gte-base, trying fallback: {e}")
            try:
                _embed_model_cache = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                logger.info("Fallback embedding model loaded: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise fallback_error
    
    return _embed_model_cache

def retrieve_docs_semantic_pinecone(user_query: str, k: int = 10) -> Dict[str, Any]:
    """
    Retrieve semantically relevant documents from Pinecone.
    Replaces the ChromaDB-based retrieval for Streamlit Cloud deployment.
    """
    try:
        # Get embedding model and Pinecone index
        embed_model = get_embed_model()
        index = get_pinecone_index()
        
        # Generate query embedding
        query_embedding = embed_model.encode([user_query])[0].tolist()
        
        # Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Extract documents and metadata in ChromaDB-compatible format
        docs = []
        
        for match in search_results.matches:
            if match.metadata and "document" in match.metadata:
                # Create ChromaDB-compatible document format
                doc = {
                    "text": match.metadata["document"],
                    "meta": match.metadata
                }
                docs.append(doc)
        
        logger.info(f"Retrieved {len(docs)} documents from Pinecone")
        
        return {
            "docs": docs,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Pinecone retrieval failed: {e}")
        return {
            "docs": [],
            "metadatas": [],
            "distances": [],
            "success": False,
            "error": str(e)
        }

def test_pinecone_connection() -> bool:
    """Test Pinecone connection and return status"""
    try:
        index = get_pinecone_index()
        stats = index.describe_index_stats()
        logger.info(f"Pinecone connection successful. Index stats: {stats}")
        return True
    except Exception as e:
        logger.error(f"Pinecone connection failed: {e}")
        return False

# Environment variable setup for Streamlit Cloud
def setup_pinecone_from_env():
    """Setup Pinecone using environment variables (for Streamlit Cloud)"""
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "sqlgen-schema-docs")
        environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        
        if not api_key:
            logger.error("PINECONE_API_KEY environment variable not set")
            return False
        
        # Create config file from environment variables
        config = {
            "pinecone_api_key": api_key,
            "pinecone_index_name": index_name,
            "pinecone_environment": environment
        }
        
        with open("pinecone_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Pinecone configuration created from environment variables")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup Pinecone from environment: {e}")
        return False
