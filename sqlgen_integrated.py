"""
Integrated SQL Generation Module
Direct integration with sqlgen.py functions (no subprocess calls)
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import sys

# Add current directory to path to import sqlgen functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import sqlgen functions directly
try:
    from sqlgen import (
        generate_sql_from_text_semantic,
        OracleSQLChain,
        LANGCHAIN_AVAILABLE
    )
    SQLGEN_AVAILABLE = True
except ImportError as e:
    SQLGEN_AVAILABLE = False
    print(f"Could not import sqlgen functions: {e}")

# Configure logging for SQL generation (separate from Streamlit)
import os
from datetime import datetime

# Create a dedicated SQL generation log file
log_file = f'sqlgen_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# Create a separate logger for SQL generation (not root logger)
logger = logging.getLogger('sqlgen_integrated')
logger.setLevel(logging.INFO)

# Clear any existing handlers for this specific logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create file handler for SQL generation logs
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("=== SQLGEN INTEGRATED LOGGING INITIALIZED ===")
logger.info(f"Log file: {log_file}")

def generate_sql_integrated(user_query: str, model: str = "gpt-4o-mini", method: str = "langchain") -> Dict[str, Any]:
    """
    Generate SQL using direct function calls (no subprocess)
    This avoids reloading the embedding model every time
    """
    try:
        logger.info(f"=== STARTING SQL GENERATION ===")
        logger.info(f"Query: {user_query}")
        logger.info(f"Method: {method}, Model: {model}")
        logger.info(f"Generating SQL with method: {method}, model: {model}")
        
        if not SQLGEN_AVAILABLE:
            return {
                "success": False,
                "data": None,
                "error": "SQL generation module not available"
            }
        
        if method == "langchain":
            if not LANGCHAIN_AVAILABLE:
                return {
                    "success": False,
                    "data": None,
                    "error": "LangChain not available. Install with: pip install langchain langchain-community langchain-openai langchain-ollama"
                }
            
            try:
                # Use LangChain approach
                sql_chain = OracleSQLChain(model_name=model)
                result = sql_chain.generate_sql_with_chromadb(user_query)
                
                if result.get("success", False):
                    return {
                        "success": True,
                        "data": result,
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "data": None,
                        "error": result.get("error", "LangChain generation failed")
                    }
                    
            except Exception as e:
                logger.error(f"LangChain generation failed: {e}")
                return {
                    "success": False,
                    "data": None,
                    "error": f"LangChain generation failed: {str(e)}"
                }
        
        else:  # semantic method
            try:
                # Use semantic approach
                result = generate_sql_from_text_semantic(user_query, model=model)
                
                if result.get("llm_sql") and not result.get("llm_sql", "").startswith("-- ERROR:"):
                    return {
                        "success": True,
                        "data": result,
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "data": None,
                        "error": result.get("error", "Semantic generation failed")
                    }
                    
            except Exception as e:
                logger.error(f"Semantic generation failed: {e}")
                return {
                    "success": False,
                    "data": None,
                    "error": f"Semantic generation failed: {str(e)}"
                }
                
    except Exception as e:
        logger.error(f"Unexpected error in SQL generation: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {str(e)}"
        }

# Test function
if __name__ == "__main__":
    # Test the integrated approach
    test_query = "Show total receipts per customer in Q2 2025"
    
    print("Testing integrated SQL generation...")
    result = generate_sql_integrated(test_query, "gpt-4o-mini", "semantic")
    
    if result["success"]:
        print("✅ Success!")
        print(f"SQL: {result['data'].get('sql', 'No SQL found')}")
    else:
        print(f"❌ Failed: {result['error']}")
