# sqlgen.py
"""
Text -> retrieve (Chroma) -> prompt -> Ollama REST API (/api/chat) -> validate (sqlglot)
Enhanced with primary transactional table detection and foreign key relationship awareness.
"""

import os
import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# Conditional ChromaDB import - only import if not in Pinecone-only mode
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available - will use Pinecone mode only")

# Removed SQLGlot dependency - using simple regex patterns for better reliability
from datetime import datetime

# OpenAI integration
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. Install with: pip install openai python-dotenv")

# LangChain integration
try:
    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain packages not available. Install with: pip install langchain langchain-community langchain-openai langchain-ollama")

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sqlgen_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
# Direct configuration values (no .env file needed)
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "schema_docs"
EMBED_MODEL_NAME = "thenlper/gte-base"
RETRIEVE_K = 150  # Increased for Pinecone mode to compensate for lack of metadata filtering
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 120  # 2 minutes for faster fallback

# OpenAI configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Keywords to identify non-transactional tables
NON_TRANSACTIONAL_KEYWORDS = ['temporary', 'temporarily', 'internal', 'audit', 'log', 'interface', 'staging', 'backup', 'history', 'archive']

# Generic column patterns to exclude (both lowercase and uppercase)
GENERIC_COLUMN_PATTERNS = [
    r'^segment\d+$', r'^attribute\d+$', r'^attribute_number\d+$',
    r'^attribute_date\d+$', r'^information\d+$', r'^char\d+$',
    r'^date\d+$', r'^number\d+$', r'^reference\d+$', r'^flexfield\d+$',
    r'^context\d+$', r'^category\d+$', r'^value\d+$', r'^text\d+$',
    r'^field\d+$', r'^column\d+$', r'^item\d+$', r'^element\d+$',
    r'^parameter\d+$', r'^variable\d+$', r'^option\d+$', r'^setting\d+$',
    r'^global_attribute$',
    # Uppercase versions
    r'^SEGMENT\d+$', r'^ATTRIBUTE\d+$', r'^ATTRIBUTE_NUMBER\d+$',
    r'^ATTRIBUTE_DATE\d+$', r'^INFORMATION\d+$', r'^CHAR\d+$',
    r'^DATE\d+$', r'^NUMBER\d+$', r'^REFERENCE\d+$', r'^FLEXFIELD\d+$',
    r'^CONTEXT\d+$', r'^CATEGORY\d+$', r'^VALUE\d+$', r'^TEXT\d+$',
    r'^FIELD\d+$', r'^COLUMN\d+$', r'^ITEM\d+$', r'^ELEMENT\d+$',
    r'^PARAMETER\d+$', r'^VARIABLE\d+$', r'^OPTION\d+$', r'^SETTING\d+$',
    r'^GLOBAL_ATTRIBUTE$'
]

# Primary table indicators - specific to your financial system
PRIMARY_TABLE_INDICATORS = [
    # Your specific table patterns
    r'^GL_JE_BATCHES$', r'^GL_JE_HEADERS$', r'^GL_JE_LINES$', r'^GL_BALANCES$',
    r'^GL_CODE_COMBINATIONS$', r'^GL_SETS_OF_BOOKS$', r'^GL_PERIODS$',
    r'^GL_DAILY_RATES$', r'^GL_IMPORT_REFERENCES$',
    r'^RA_CUSTOMER_TRX_ALL$', r'^AR_PAYMENT_SCHEDULES_ALL$', r'^RA_CUSTOMER_TRX_LINES_ALL$',
    r'^AR_RECEIVABLE_APPLICATIONS_ALL$', r'^RA_CUST_TRX_LINE_GL_DIST_ALL$', r'^AR_RECEIVABLES_TRX_ALL$',
    r'^AP_INVOICES_ALL$', r'^AP_PAYMENT_SCHEDULES_ALL$', r'^AP_INVOICE_DISTRIBUTIONS_ALL$',
    r'^AP_AE_HEADERS_ALL$', r'^AP_AE_LINES_ALL$', r'^AP_HOLDS_ALL$',
    r'^FA_ADDITIONS_B$', r'^FA_BOOKS$', r'^FA_DEPRN_DETAIL$', r'^FA_DEPRN_SUMMARY$',
    r'^FA_CATEGORIES_B$', r'^FA_DEPRN_PERIODS$',
    r'^FND_ID_FLEX_SEGMENTS$', r'^FND_FLEX_VALUES$', r'^FND_ID_FLEX_STRUCTURES$',
    r'^FND_FLEX_VALUE_HIERARCHIES$', r'^FND_ID_FLEXS$',
    
    # General patterns for primary tables
    r'^(AP|AR|GL|PO|OE|INV|FA|FND)_[A-Z_]+(?!_ALL_|_V|_S|_VW|_MV|_SUMMARY|_HISTORY|_ARCHIVE|_TEMP|_BKP)',
    r'^(AP|AR|GL|PO|OE|INV|FA|FND)_[A-Z_]+_ALL$',
    r'^RA_[A-Z_]+_ALL$',
    r'^XLA_[A-Z_]+$',
    
    # Column patterns in primary tables
    r'.*TRX_ID.*', r'.*INVOICE_ID.*', r'.*PAYMENT_ID.*', r'.*ORDER_ID.*',
    r'.*BATCH_ID.*', r'.*JOURNAL_ID.*', r'.*LEDGER_ID.*', r'.*CUSTOMER_ID.*',
    r'.*VENDOR_ID.*', r'.*EMPLOYEE_ID.*', r'.*ASSET_ID.*', r'.*CODE_COMBINATION_ID.*',
    
    # Foreign key patterns
    r'.*_ID$', r'.*_CODE$', r'.*_NUM$', r'.*_NUMBER$',
    
    # Date patterns in transactional tables
    r'.*_DATE$', r'.*_DT$', r'CREATION_DATE', r'LAST_UPDATE_DATE',
    
    # Amount patterns
    r'.*_AMOUNT$', r'.*_AMT$', r'.*_VALUE$', r'.*_COST$', r'.*_PRICE$'
]

SUMMARY_TABLE_INDICATORS = [
    # Table name patterns for summary tables
    r'.*_SUMMARY$', r'.*_HISTORY$', r'.*_ARCHIVE$', r'.*_TEMP$', r'.*_BKP$',
    r'.*_V$', r'.*_S$', r'.*_VW$', r'.*_MV$', r'.*_ROLLUP$', r'.*_AGG$',
    r'.*_CONS$', r'.*_BALANCE$', r'.*_STATS$', r'.*_REPORT$',
    
    # Column patterns in summary tables
    r'.*_COUNT$', r'.*_TOTAL$', r'.*_AVG$', r'.*_SUM$', r'.*_MIN$', r'.*_MAX$',
    r'.*_BALANCE$', r'.*_VALUE$', r'.*_AMOUNT$',
    
    # Comments indicating summary tables
    'summary', 'consolidated', 'history', 'archive', 'backup', 'temporary',
    'staging', 'reporting', 'rollup', 'aggregate', 'snapshot', 'balance',
    'statistics', 'audit', 'log'
]

# Module scores for prioritization
MODULE_SCORES = {
    'GL': 2.0,  # General Ledger - highest priority
    'AR': 2.0,  # Accounts Receivable - highest priority  
    'AP': 2.0,  # Accounts Payable - highest priority
    'FA': 1.8,  # Fixed Assets - high priority
    'FND': 1.5, # Application Object Library - medium priority
    'PO': 1.0, 'OE': 1.0, 'INV': 1.0, 'XLA': 1.0  # Other modules
}

# ---------- init models / chroma ----------
# Global model cache to avoid reloading
_embed_model_cache = None
_chroma_client_cache = None
_chroma_collection_cache = None
_primary_tables_cache = None
_process_context_cache = {}

def get_embed_model():
    """Get cached embedding model or load it once with explicit caching and PyTorch compatibility"""
    global _embed_model_cache
    if _embed_model_cache is None:
        import os
        # Create cache directory if it doesn't exist
        cache_dir = os.path.expanduser("~/.cache/sentence_transformers")
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        
        try:
            # Use explicit cache directory with PyTorch compatibility settings
            _embed_model_cache = SentenceTransformer(
                EMBED_MODEL_NAME, 
                cache_folder=cache_dir,
                device='cpu',  # Force CPU to avoid device issues
                trust_remote_code=True  # Allow remote code for compatibility
            )
            logger.info("Embedding model loaded and cached")
            
        except Exception as e:
            logger.warning(f"Failed to load {EMBED_MODEL_NAME}, trying fallback model: {e}")
            try:
                # Fallback to a more stable model
                _embed_model_cache = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    cache_folder=cache_dir,
                    device='cpu'
                )
                logger.info("Fallback embedding model loaded: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise fallback_error
                
    return _embed_model_cache

def get_dynamic_primary_tables() -> List[str]:
    """
    Dynamically extract primary transactional tables from process_docs folder.
    This replaces hardcoded PRIMARY_TABLE_INDICATORS with business-driven detection.
    """
    global _primary_tables_cache
    
    if _primary_tables_cache is not None:
        return _primary_tables_cache
    
    import re
    from pathlib import Path
    
    primary_tables = set()
    
    # Read process documentation
    process_docs_dir = Path("process_docs")
    if not process_docs_dir.exists():
        logger.warning("process_docs directory not found, using fallback primary tables")
        # Fallback to essential tables if process_docs not available
        _primary_tables_cache = [
            'GL_JE_HEADERS', 'GL_JE_LINES', 'GL_JE_BATCHES', 'GL_BALANCES',
            'RA_CUSTOMER_TRX_ALL', 'AR_PAYMENT_SCHEDULES_ALL', 'RA_CUSTOMER_TRX_LINES_ALL',
            'AP_INVOICES_ALL', 'AP_PAYMENT_SCHEDULES_ALL', 'AP_INVOICE_DISTRIBUTIONS_ALL',
            'FA_ADDITIONS_B', 'FA_BOOKS', 'FA_DEPRN_DETAIL', 'FA_DEPRN_SUMMARY'
        ]
        return _primary_tables_cache
    
    for process_file in process_docs_dir.glob("*.txt"):
        try:
            content = process_file.read_text(encoding='utf-8')
            
            # Extract tables mentioned in business processes
            table_patterns = [
                r'Tables:\s*([^-\n]+)',  # "Tables: GL_JE_HEADERS, GL_JE_LINES, GL_JE_BATCHES"
                r'Table:\s*([A-Z_]+)',   # "Table: GL_JE_HEADERS"
                r'([A-Z_]+_ALL?)\s*\(',  # Table names in parentheses
                r'([A-Z_]+_[A-Z_]+)\s*-', # Table names followed by dash
            ]
            
            for pattern in table_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, str):
                        # Handle comma-separated tables
                        tables = [t.strip() for t in match.split(',')]
                        for table in tables:
                            # Clean up table names
                            table = table.strip().upper()
                            if table and '_' in table and len(table) > 3:
                                primary_tables.add(table)
            
            logger.debug(f"Extracted {len(primary_tables)} primary tables from {process_file.name}")
            
        except Exception as e:
            logger.warning(f"Error reading process file {process_file}: {e}")
    
    # Convert to sorted list for consistency
    _primary_tables_cache = sorted(list(primary_tables))
    logger.info(f"Dynamic primary tables extracted: {len(_primary_tables_cache)} tables")
    logger.debug(f"Primary tables: {_primary_tables_cache[:10]}...")
    
    return _primary_tables_cache

def get_dynamic_skip_validation_list() -> List[str]:
    """
    Dynamically generate skip validation list based on SQL functions and common patterns.
    This replaces hardcoded validation patterns with dynamic detection.
    """
    # SQL Keywords (always skip)
    sql_keywords = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'BETWEEN', 'IN', 'EXISTS',
        'GROUP', 'BY', 'ORDER', 'HAVING', 'UNION', 'ALL', 'DISTINCT', 'AS', 'IS', 'NULL',
        'NOT', 'LIKE', 'ESCAPE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IF'
    }
    
    # SQL Functions (always skip)
    sql_functions = {
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'TRUNC', 'SUBSTR', 'UPPER', 'LOWER',
        'LENGTH', 'REPLACE', 'DECODE', 'COALESCE', 'NVL', 'TO_CHAR', 'TO_NUMBER', 'TO_DATE',
        'CAST', 'CONVERT', 'EXTRACT', 'SYSDATE', 'CURRENT_DATE', 'CURRENT_TIMESTAMP',
        'ROWID', 'ROWNUM', 'LEVEL', 'CONNECT_BY_ROOT', 'PRIOR', 'FETCH', 'FIRST', 'ROWS',
        'ONLY', 'ROW', 'DESC', 'ASC'
    }
    
    # Date/Time Format Elements (always skip)
    date_formats = {
        'YYYY', 'MM', 'DD', 'HH24', 'MI', 'SS', 'MONTH', 'YEAR', 'DAY', 'HH', 'AM', 'PM'
    }
    
    # Common SQL Aliases (dynamically detected patterns)
    common_aliases = set()
    
    # Add common aggregate aliases
    aggregate_patterns = [
        'TOTAL_', 'SUM_', 'AVG_', 'MAX_', 'MIN_', 'COUNT_', 'BALANCE_', 'AMOUNT_',
        'UNPAID_', 'PAID_', 'REMAINING_', 'OUTSTANDING_', 'APPLIED_', 'DUE_'
    ]
    
    # Add common entity aliases
    entity_patterns = [
        'SUPPLIER_', 'CUSTOMER_', 'VENDOR_', 'INVOICE_', 'PAYMENT_', 'TRANSACTION_',
        'ORDER_', 'RECEIPT_', 'DOCUMENT_', 'BATCH_', 'HEADER_', 'LINE_'
    ]
    
    # Add common attribute aliases
    attribute_patterns = [
        '_NAME', '_NUMBER', '_DATE', '_AMOUNT', '_ID', '_CODE', '_STATUS', '_TYPE'
    ]
    
    # Generate common aliases dynamically
    for prefix in aggregate_patterns:
        for suffix in ['AMOUNT', 'COUNT', 'BALANCE', 'VALUE', 'TOTAL']:
            common_aliases.add(f"{prefix}{suffix}")
    
    for prefix in entity_patterns:
        for suffix in ['NAME', 'NUMBER', 'DATE', 'AMOUNT', 'ID', 'CODE']:
            common_aliases.add(f"{prefix}{suffix}")
    
    for prefix in ['INVOICE', 'PAYMENT', 'TRANSACTION', 'ORDER', 'RECEIPT', 'DOCUMENT']:
        for suffix in attribute_patterns:
            common_aliases.add(f"{prefix}{suffix}")
    
    # Combine all skip patterns
    skip_validation = list(sql_keywords | sql_functions | date_formats | common_aliases)
    
    logger.debug(f"Generated dynamic skip validation list with {len(skip_validation)} items")
    return skip_validation

def get_chroma_client():
    """Get cached ChromaDB client or create it once"""
    global _chroma_client_cache
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available - use Pinecone mode instead")
    if _chroma_client_cache is None:
        logger.info(f"Connecting to ChromaDB: {CHROMA_PERSIST_DIR}")
        _chroma_client_cache = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info("ChromaDB client created and cached")
    return _chroma_client_cache

def get_chroma_collection():
    """Get cached ChromaDB collection or get it once"""
    global _chroma_collection_cache
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available - use Pinecone mode instead")
    if _chroma_collection_cache is None:
        client = get_chroma_client()
        logger.info(f"Getting ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        _chroma_collection_cache = client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info("ChromaDB collection retrieved and cached")
    return _chroma_collection_cache

# Lazy initialization - only load when actually needed
embed_model = None
client = None
collection = None

def get_global_embed_model():
    """Get the global embed model, initializing if needed"""
    global embed_model
    if embed_model is None:
        embed_model = get_embed_model()
    return embed_model

def get_global_chroma_client():
    """Get the global chroma client, initializing if needed"""
    global client
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available - use Pinecone mode instead")
    if client is None:
        client = get_chroma_client()
    return client

def get_global_chroma_collection():
    """Get the global chroma collection, initializing if needed"""
    global collection
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("ChromaDB not available - use Pinecone mode instead")
    if collection is None:
        collection = get_chroma_collection()
    return collection

# Initialize OpenAI client if available
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized with model: {OPENAI_MODEL}")
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {e}")
        openai_client = None
else:
    logger.info("OpenAI not available - using Ollama only")

def get_full_table_comment(table_name: str, doc_text: str) -> str:
    """
    Extract the full table comment from a document, handling chunked comments.
    """
    if not doc_text or not table_name:
        return ""
    
    # Look for TABLE_SUMMARY in the document
    if "TABLE_SUMMARY:" in doc_text:
        # Extract the table summary section
        summary_section = doc_text.split("TABLE_SUMMARY:")[1]
        # Get the first line (which should contain the main description)
        first_line = summary_section.split("\n")[0].strip()
        return first_line
    
    return ""

def is_non_transactional_table(table_comment):
    """Check if a table is non-transactional based on its comment"""
    if not table_comment:
        logger.debug("No table comment provided, not filtering")
        return False
        
    table_comment_lower = table_comment.lower()
    logger.debug(f"Checking table comment: '{table_comment_lower[:200]}...'")
    
    # Check for non-transactional keywords
    for keyword in NON_TRANSACTIONAL_KEYWORDS:
        if keyword in table_comment_lower:
            logger.info(f"Found non-transactional keyword '{keyword}' in comment")
            return True
    
    # Additional patterns for temporary/summary tables
    temporary_patterns = [
        'temporary table', 'temp table', 'staging table', 'summary table',
        'consolidated table', 'rollup table', 'aggregate table', 'snapshot table',
        'backup table', 'archive table', 'history table', 'audit table',
        'log table', 'interface table', 'selected for payment', 'payment batch',
        'selected invoices', 'selected invoice', 'batch processing'
    ]
    
    for pattern in temporary_patterns:
        if pattern in table_comment_lower:
            logger.info(f"Found temporary pattern '{pattern}' in comment")
            return True
    
    logger.debug("Table comment does not match non-transactional patterns")
    return False

def is_generic_column(column_name):
    """Check if a column name matches generic patterns"""
    if not column_name:
        return False
        
    column_name_lower = column_name.lower()
    column_name_upper = column_name.upper()
    
    for pattern in GENERIC_COLUMN_PATTERNS:
        if re.match(pattern, column_name_lower) or re.match(pattern, column_name_upper):
            return True
    return False

def embed_query(text: str):
    model = get_global_embed_model()
    vec = model.encode([text], show_progress_bar=False)[0]
    return vec.tolist()

def extract_semantic_meaning(text, max_length=200):
    """
    Extract the core semantic meaning from table/column comments,
    focusing on the purpose and content rather than technical details.
    """
    if not text:
        return ""
    
    # Remove technical prefixes and focus on descriptive content
    patterns_to_remove = [
        r'TABLE: \w+\n',
        r'COLUMN: \w+\n',
        r'TYPE: [^\n]+\n',
        r'PRIMARY_KEY: (YES|NO)\n',
        r'FOREIGN KEYS: [^\n]+\n',
        r'INDEXES: [^\n]+\n',
        r'COLUMNS: [^\n]+\n',
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text)
    
    # Extract the most meaningful part (usually the first sentence)
    sentences = re.split(r'[.!?]+', text)
    meaningful_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Skip technical or generic sentences
        if any(term in sentence.lower() for term in [
            'who column', 'oracle internal use', 'used to implement',
            'object version number', 'identifier for', 'flag for',
            'sequential numbering', 'reference column'
        ]):
            continue
            
        # Prioritize sentences that describe purpose or content
        if any(indicator in sentence.lower() for indicator in [
            'stores', 'contains', 'holds', 'manages', 'tracks', 'records',
            'for example', 'such as', 'including', 'related to'
        ]):
            meaningful_sentences.append(sentence)
        else:
            meaningful_sentences.append(sentence)
    
    # Combine the most meaningful parts
    result = '. '.join(meaningful_sentences[:3])  # Take up to 3 most meaningful sentences
    if len(result) > max_length:
        result = result[:max_length] + '...'
    
    return result

def extract_primary_key(comments):
    """
    Extract primary key information from table comments.
    """
    primary_key = []
    for comment in comments:
        if 'PRIMARY KEY:' in comment:
            # Extract the primary key section
            pk_section = comment.split('PRIMARY KEY:')[1].split('\n')[0].strip()
            primary_key = [col.strip() for col in pk_section.split(',')]
            break
    return primary_key

def extract_foreign_keys(comments):
    """
    Extract foreign key relationships from table comments.
    """
    foreign_keys = []
    for comment in comments:
        if 'FOREIGN KEYS:' in comment:
            # Extract the foreign key section
            fk_section = comment.split('FOREIGN KEYS:')[1].split('\n')[0].strip()
            # Parse foreign key relationships
            fk_matches = re.findall(r'(\w+):(\w+)\s*->\s*(\w+)\((\w+)\)', fk_section)
            for match in fk_matches:
                foreign_keys.append({
                    'constraint_name': match[0],
                    'source_column': match[1],
                    'references_table': match[2],
                    'references_column': match[3]
                })
            break
    return foreign_keys

def extract_financial_terms(user_query: str) -> List[str]:
    """
    Extract financial and business terms from user query for table candidate identification.
    """
    query_lower = user_query.lower()
    financial_terms = []
    
    # Invoice/Payment terms
    if any(term in query_lower for term in ['invoice', 'invoices', 'billing']):
        financial_terms.extend(['invoice', 'payment', 'receivable'])
    if any(term in query_lower for term in ['payment', 'payments', 'paid', 'unpaid']):
        financial_terms.extend(['payment', 'receivable', 'invoice'])
    if any(term in query_lower for term in ['due', 'overdue', 'past due']):
        financial_terms.extend(['due', 'payment', 'receivable'])
    
    # General Ledger terms
    if any(term in query_lower for term in ['journal', 'journals', 'entry', 'entries']):
        financial_terms.extend(['journal', 'ledger', 'account'])
    if any(term in query_lower for term in ['ledger', 'account', 'accounts', 'balance']):
        financial_terms.extend(['ledger', 'account', 'journal'])
    if any(term in query_lower for term in ['gl', 'general ledger']):
        financial_terms.extend(['gl', 'ledger', 'journal'])
    
    # Asset terms
    if any(term in query_lower for term in ['asset', 'assets', 'depreciation', 'depreciate']):
        financial_terms.extend(['asset', 'depreciation', 'fixed'])
    if any(term in query_lower for term in ['fixed asset', 'fixed assets']):
        financial_terms.extend(['fixed', 'asset', 'depreciation'])
    
    # Customer/Vendor terms
    if any(term in query_lower for term in ['customer', 'customers', 'client', 'clients']):
        financial_terms.extend(['customer', 'receivable'])
    if any(term in query_lower for term in ['vendor', 'vendors', 'supplier', 'suppliers']):
        financial_terms.extend(['vendor', 'payable'])
    
    return list(set(financial_terms))

def get_primary_table_candidates(user_query: str) -> List[str]:
    """
    Pre-filter to identify primary table candidates before semantic retrieval.
    This reduces noise and improves retrieval quality.
    """
    financial_terms = extract_financial_terms(user_query)
    primary_candidates = []
    
    # Map financial terms to likely primary tables
    for term in financial_terms:
        if term in ['invoice', 'payment', 'receivable']:
            primary_candidates.extend([
                'AP_INVOICES_ALL', 'AR_CUSTOMER_TRX_ALL', 'AP_PAYMENT_SCHEDULES_ALL',
                'AR_PAYMENT_SCHEDULES_ALL', 'RA_CUSTOMER_TRX_LINES_ALL',
                'AR_RECEIVABLE_APPLICATIONS_ALL', 'AP_INVOICE_DISTRIBUTIONS_ALL'
            ])
        elif term in ['journal', 'ledger', 'account', 'gl']:
            primary_candidates.extend([
                'GL_JE_HEADERS', 'GL_JE_LINES', 'GL_JE_BATCHES', 'GL_BALANCES',
                'GL_CODE_COMBINATIONS', 'GL_SETS_OF_BOOKS', 'GL_PERIODS'
            ])
        elif term in ['asset', 'depreciation', 'fixed']:
            primary_candidates.extend([
                'FA_ADDITIONS_B', 'FA_BOOKS', 'FA_DEPRN_DETAIL', 'FA_DEPRN_SUMMARY',
                'FA_CATEGORIES_B', 'FA_DEPRN_PERIODS'
            ])
        elif term in ['customer']:
            primary_candidates.extend([
                'AR_CUSTOMER_TRX_ALL', 'RA_CUSTOMER_TRX_LINES_ALL', 'AR_RECEIVABLE_APPLICATIONS_ALL'
            ])
        elif term in ['vendor']:
            primary_candidates.extend([
                'AP_INVOICES_ALL', 'AP_PAYMENT_SCHEDULES_ALL', 'AP_INVOICE_DISTRIBUTIONS_ALL'
            ])
    
    # If no specific terms found, include all known primary tables
    if not primary_candidates:
        primary_candidates = [
            'GL_JE_BATCHES', 'GL_JE_HEADERS', 'GL_JE_LINES', 'GL_BALANCES',
            'GL_CODE_COMBINATIONS', 'GL_SETS_OF_BOOKS', 'GL_PERIODS',
            'RA_CUSTOMER_TRX_ALL', 'AR_PAYMENT_SCHEDULES_ALL', 'RA_CUSTOMER_TRX_LINES_ALL',
            'AR_RECEIVABLE_APPLICATIONS_ALL', 'AP_INVOICES_ALL', 'AP_PAYMENT_SCHEDULES_ALL',
            'AP_INVOICE_DISTRIBUTIONS_ALL', 'FA_ADDITIONS_B', 'FA_BOOKS', 'FA_DEPRN_DETAIL'
        ]
    
    return list(set(primary_candidates))

def expand_with_related_tables(primary_tables: List[str]) -> List[str]:
    """
    Find tables related to primary tables through PK-FK relationships.
    """
    related_tables = set(primary_tables)
    
    try:
        for table in primary_tables:
            # Get foreign key relationships for this table
            collection = get_global_chroma_collection()
            fk_results = collection.query(
                query_texts=[f"foreign keys for table {table}"],
                n_results=50,
                where={"$and": [{"table": table}, {"doc_type": "table"}]}
            )
            
            # Extract referenced tables from FK relationships
            if fk_results and fk_results['documents']:
                for doc in fk_results['documents'][0]:
                    if "FOREIGN KEYS:" in doc:
                        # Parse FK relationships to find referenced tables
                        fk_matches = re.findall(r'->\s*(\w+)\(', doc)
                        for ref_table in fk_matches:
                            if is_primary_transactional_table(ref_table, {}):
                                related_tables.add(ref_table)
                                logger.debug(f"Found related table via FK: {ref_table} -> {table}")
    except Exception as e:
        logger.warning(f"Could not expand related tables: {e}")
    
    return list(related_tables)


# Simplified but more effective primary table detection
def is_primary_transactional_table(table_name, table_info):
    """
    Dynamic primary table detection using business process documentation.
    """
    table_name_upper = table_name.upper()
    
    # 1. FIRST: Check if this is one of the dynamically extracted primary tables
    primary_tables = get_dynamic_primary_tables()
    
    if table_name_upper in primary_tables:
        return True
    
    # 2. Check for summary table patterns (exclude these)
    summary_patterns = [
        r'.*_SUMMARY$', r'.*_HISTORY$', r'.*_ARCHIVE$', r'.*_TEMP$', r'.*_BKP$',
        r'.*_V$', r'.*_S$', r'.*_VW$', r'.*_MV$', r'.*_ROLLUP$', r'.*_AGG$',
        r'.*_CONS$', r'.*_BALANCE$', r'.*_STATS$', r'.*_REPORT$', r'.*_INTERIM$'
    ]
    
    for pattern in summary_patterns:
        if re.search(pattern, table_name_upper):
            return False
    
    # 3. Check module prefix
    if table_name_upper.startswith(('GL_', 'AR_', 'AP_', 'FA_', 'FND_')):
        return True
    
    return False




def calculate_table_priority_score(table_name, table_info, user_query: str = ""):
    """
    Semantic priority scoring using table comments, column descriptions, and process context.
    NO hardcoded table names or keywords - purely semantic.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    score = 0.0
    
    # Load embedding model for semantic similarity
    try:
        model = get_embed_model()
    except Exception as e:
        logger.warning(f"Could not load embedding model for priority scoring: {e}")
        return 10.0  # Fallback score
    
    # Get table comment for semantic analysis
    table_comment = table_info.get("table_comment", "")
    if not table_comment:
        # Try to get from comments list
        for comment in table_info.get("comments", []):
            if "TABLE_SUMMARY:" in comment:
                table_comment = comment.split("TABLE_SUMMARY:")[1].split("\n")[0].strip()
                break
    
    # Calculate semantic similarity between user query and table comment
    if user_query and table_comment:
        try:
            query_embedding = model.encode([user_query])
            table_embedding = model.encode([table_comment])
            semantic_similarity = np.dot(query_embedding, table_embedding.T).flatten()[0]
            score += semantic_similarity * 20.0  # Weight semantic similarity heavily
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
    
    # ENHANCEMENT: Add process context bonus
    # Get process context for this query to boost tables relevant to the business process
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        
        # Search for relevant process documentation
        query_embedding = model.encode([user_query]).tolist()[0]
        try:
            process_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"document_type": "business_process"}
            )
        except:
            # Fallback: search all documents and filter by content
            process_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            # Filter results to only include process documents
            if process_results['documents'] and process_results['documents'][0]:
                filtered_docs = []
                for i, doc in enumerate(process_results['documents'][0]):
                    if doc and ("PROCESS:" in doc or "OVERVIEW:" in doc or "STEPS:" in doc):
                        filtered_docs.append(doc)
                process_results = {
                    'documents': [filtered_docs[:2]]  # Limit to top 2
                }
        
        if process_results['documents'] and process_results['documents'][0]:
            # Check if this table is mentioned in the process context
            process_context = " ".join(process_results['documents'][0])
            if table_name.upper() in process_context.upper():
                score += 15.0  # Significant bonus for tables mentioned in process context
                logger.info(f"Process context bonus applied to {table_name}")
    except Exception as e:
        logger.debug(f"Could not apply process context bonus: {e}")
    
    # Bonus for tables with rich column descriptions
    column_descriptions = []
    for comment in table_info.get("comments", []):
        if "DESCRIPTION:" in comment:
            desc = comment.split("DESCRIPTION:")[1].split("\n")[0].strip()
            if desc:
                column_descriptions.append(desc)
    
    if column_descriptions:
        score += min(len(column_descriptions) * 0.5, 5.0)  # Bonus for rich descriptions
    
    # PK-FK relationship bonus (this is structural, not semantic)
    if table_info.get('foreign_keys'):
        score += 2.0  # Tables with relationships are more important
    
    # Ensure minimum score
    return max(score, 1.0)


def retrieve_docs_semantic(user_query: str, k: int = RETRIEVE_K) -> Dict[str, Any]:
    """
    Enhanced retrieval with pre-filtering and PK-FK relationship expansion.
    Uses Pinecone for Streamlit Cloud deployment, falls back to ChromaDB for local development.
    """
    logger.info(f"Enhanced semantic retrieval for query: {user_query}")
    
    # Check if we're in production (Streamlit Cloud) or local development
    if os.getenv("PINECONE_API_KEY"):
        logger.info("Using Pinecone for vector storage (production mode)")
        try:
            from sqlgen_pinecone import retrieve_docs_semantic_pinecone
            return retrieve_docs_semantic_pinecone(user_query, k)
        except ImportError:
            logger.error("Pinecone integration not available in production mode - cannot fallback to ChromaDB")
            return {
                "docs": [],
                "success": False,
                "error": "Pinecone integration not available in production mode"
            }
        except Exception as e:
            logger.error(f"Pinecone retrieval failed in production mode: {e}")
            return {
                "docs": [],
                "success": False,
                "error": f"Pinecone retrieval failed: {e}"
            }
    
    # Local development mode - use ChromaDB
    logger.info("Using ChromaDB for vector storage (local development mode)")
    
    # Check if ChromaDB is available
    if not CHROMADB_AVAILABLE:
        logger.error("ChromaDB not available in local development mode")
        return {
            "docs": [],
            "success": False,
            "error": "ChromaDB not available - install ChromaDB or use Pinecone mode"
        }
    
    # CRITICAL: Double-check we're actually in local development
    if os.getenv("PINECONE_API_KEY"):
        logger.error("CRITICAL ERROR: PINECONE_API_KEY is set but Pinecone failed - cannot fallback to ChromaDB in production")
        return {
            "docs": [],
            "success": False,
            "error": "Production mode requires Pinecone - ChromaDB fallback not available"
        }
    
    # 1. Get primary table candidates based on query terms
    primary_candidates = get_primary_table_candidates(user_query)
    logger.info(f"Identified {len(primary_candidates)} primary table candidates: {primary_candidates[:5]}...")
    
    # 2. Expand with PK-FK related tables
    related_tables = expand_with_related_tables(primary_candidates)
    logger.info(f"Expanded to {len(related_tables)} related tables via PK-FK relationships")
    
    # 3. Embed the user query
    q_emb = embed_query(user_query)
    
    # 4. Retrieve with table filtering (if we have candidates)
    if primary_candidates:
        try:
            # Try filtered retrieval first
            # Simplified query to avoid ChromaDB operator issues
            logger.info(f"Filtered retrieval with {len(related_tables)} tables")
            collection = get_global_chroma_collection()
            res = collection.query(
                query_embeddings=[q_emb], 
                n_results=k*2,  # Reduced since we're pre-filtering
                where={"table": {"$in": list(related_tables)}},
                include=["documents", "metadatas", "distances"]
            )
            logger.info(f"Filtered retrieval found {len(res['documents'][0])} documents")
        except Exception as e:
            logger.warning(f"Filtered retrieval failed: {e}, falling back to full retrieval")
            res = collection.query(
                query_embeddings=[q_emb], 
                n_results=k*3,
                include=["documents", "metadatas", "distances"]
            )
    else:
        # Fallback to full retrieval
        collection = get_global_chroma_collection()
        res = collection.query(
            query_embeddings=[q_emb], 
            n_results=k*3,
        include=["documents", "metadatas", "distances"]
    )
    
    docs = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        # CRITICAL: Filter out non-transactional tables early in retrieval
        table_name = meta.get("table")
        # Get table comment - prioritize full document comment over metadata
        full_comment = get_full_table_comment(table_name, doc)
        
        # If no full comment found, try metadata
        if not full_comment:
            full_comment = meta.get("table_comment", "")
            logger.debug(f"Using metadata table comment: '{full_comment}'")
        
        # If still no comment, try to extract from document content
        if not full_comment and "TABLE_SUMMARY:" in doc:
            full_comment = doc.split("TABLE_SUMMARY:")[1].split("\n")[0].strip()
            logger.debug(f"Extracted table comment from document: '{full_comment}'")
        
        logger.debug(f"Final table comment for {table_name}: '{full_comment}'")
        
        if table_name and is_non_transactional_table(full_comment):
            logger.info(f"FILTERING OUT non-transactional table during retrieval: {table_name} (comment: {full_comment[:100]}...)")
            continue
        
        # Extract semantic meaning from the document
        semantic_text = extract_semantic_meaning(doc)
        if not semantic_text:
            continue
            
        # Calculate semantic similarity
        semantic_emb = embed_query(semantic_text)
        semantic_sim = cosine_similarity([q_emb], [semantic_emb])[0][0]
        
        # Calculate enhanced table priority score
        table_priority = 0.0
        if meta.get("doc_type") == "table" and table_name:
            # Create a mock table_info for priority calculation
            table_info = {"comments": [doc], "columns": []}
            # Try to extract columns from the document text
            for line in doc.split('\n'):
                if line.strip().upper().startswith('COLUMNS:'):
                    cols = line.split(':', 1)[1].strip()
                    table_info["columns"] = [c.strip() for c in cols.split(',')]
                    break
            table_priority = calculate_table_priority_score(table_name, table_info, user_query)
        
        # Enhanced scoring with query-specific weighting
        # Normalize table_priority to 0-1 range (assuming max priority is around 200)
        normalized_priority = min(table_priority / 200.0, 1.0)
        combined_score = (0.25 * (1 - dist)) + (0.45 * semantic_sim) + (0.30 * normalized_priority)
        
        docs.append({
            "text": doc,
            "semantic_text": semantic_text,
            "meta": meta,
            "distance": dist,
            "semantic_similarity": semantic_sim,
            "table_priority": table_priority,
            "combined_score": combined_score
        })
    
    # Sort by combined score (higher is better)
    docs.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Enhanced selection with better primary table prioritization
    primary_docs = [doc for doc in docs if doc.get("table_priority", 0) > 10]  # Primary tables should have high priority scores
    other_docs = [doc for doc in docs if doc.get("table_priority", 0) <= 10]
    
    # Prioritize primary tables but include some others for context
    selected_docs = primary_docs[:min(k, len(primary_docs))] 
    remaining_slots = k - len(selected_docs)
    if remaining_slots > 0:
        selected_docs.extend(other_docs[:remaining_slots])
    
    logger.info(f"Retrieved {len(selected_docs)} documents with enhanced filtering and priority scoring")
    logger.info(f"Primary tables found: {len(primary_docs)}, Other tables: {len(other_docs)}")
    
    for i, doc in enumerate(selected_docs):
        logger.debug(f"Doc {i+1}: Score={doc['combined_score']:.3f}, Priority={doc['table_priority']:.1f}, Table={doc['meta'].get('table', 'N/A')}, Semantic: {doc['semantic_text'][:100]}...")
    
    return {"docs": selected_docs}

def summarize_relevant_tables(docs: List[Dict[str,Any]], original_query: str = "") -> Dict[str, Dict]:
    logger.info("Summarizing relevant tables from retrieved documents")
    tables = {}
    
    # Get ChromaDB collection for semantic column search
    collection = None
    try:
        # Safety check: Only use ChromaDB in local development
        if os.getenv("PINECONE_API_KEY"):
            logger.warning("Pinecone mode detected - skipping ChromaDB column retrieval")
        elif not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - skipping ChromaDB column retrieval")
        else:
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
            logger.info("Successfully connected to ChromaDB for column retrieval")
    except Exception as e:
        logger.warning(f"Could not access ChromaDB collection: {e}")
        collection = None
    
    for d in docs:
        meta = d.get("meta", {})
        doc_type = meta.get("doc_type")
        table = meta.get("table")
        column = meta.get("column")
        table_comment = meta.get("table_comment", "")
        
        if table:
            # CRITICAL: Filter out non-transactional tables (temporary, summary, etc.)
            # Handle chunked comments by checking the full document content
            full_comment = get_full_table_comment(table, d.get("text", ""))
            if not full_comment:
                full_comment = table_comment  # Fallback to metadata comment
            
            if is_non_transactional_table(full_comment):
                logger.info(f"FILTERING OUT non-transactional table: {table} (comment: {full_comment[:100]}...)")
                continue
            
            tables.setdefault(table, {"columns": set(), "comments": [], "doc_ids": [], "table_comment": table_comment})
            
        # Extract FK relationships from any document that contains them
        txt = d.get("text", "")
        if txt and table:
            # Extract primary key and foreign keys from the current document
            primary_key = extract_primary_key([txt])
            foreign_keys = extract_foreign_keys([txt])
            
            # If no FK relationships found in current document, try to fetch from doc_type: 'table' document
            if not foreign_keys and collection:
                logger.info(f"No FK relationships found in current document for {table}, trying to fetch from doc_type: 'table' document")
                try:
                    # Query for doc_type: 'table' document for this table
                    table_results = collection.query(
                        query_texts=[f"table {table}"],
                        n_results=5,
                        where={"$and": [{"table": table.upper()}, {"doc_type": "table"}]}
                    )
                    
                    if table_results and table_results['documents']:
                        logger.info(f"Found {len(table_results['documents'][0])} documents for table {table}")
                        # Look for doc_type: 'table' document
                        for i, doc_text in enumerate(table_results['documents'][0]):
                            doc_type = table_results['metadatas'][0][i].get('doc_type')
                            logger.info(f"Checking document {i} with doc_type: {doc_type}")
                            if doc_type == 'table':
                                logger.info(f"Found doc_type: 'table' document for {table}, extracting FK relationships")
                                # Extract FK relationships from this document
                                fk_from_table_doc = extract_foreign_keys([doc_text])
                                logger.info(f"Extracted {len(fk_from_table_doc)} FK relationships from doc_type: 'table' document")
                                if fk_from_table_doc:
                                    foreign_keys = fk_from_table_doc
                                    logger.info(f"Found FK relationships for {table} in doc_type: 'table' document: {fk_from_table_doc}")
                                break
                    else:
                        logger.warning(f"No documents found for table {table}")
                except Exception as e:
                    logger.warning(f"Could not fetch FK relationships for table {table}: {e}")
            else:
                logger.info(f"FK relationships already found for {table}: {len(foreign_keys)} relationships")
            
            # Store FK relationships for this table
            if table not in tables:
                tables[table] = {"columns": set(), "comments": [], "doc_ids": [], "table_comment": table_comment}
            
            tables[table]["primary_key"] = primary_key
            tables[table]["foreign_keys"] = foreign_keys
            
            # best-effort parse column list if included
            for line in txt.splitlines():
                if line.strip().upper().startswith("COLUMNS:"):
                    cols = line.split(":",1)[1].strip()
                    for c in cols.split(","):
                        col = c.strip()
                        if col and not is_generic_column(col):
                            tables[table]["columns"].add(col)
            tables[table]["comments"].append(d.get("text",""))
            tables[table]["doc_ids"].append(meta.get("doc_id",""))
            
        elif doc_type == "column" or column:
            # Skip generic columns
            if is_generic_column(column):
                logger.debug(f"Skipping generic column: {column}")
                continue
                
            colname = column or meta.get("column")
            if colname:
                tables.setdefault(table, {"columns": set(), "comments": [], "doc_ids": [], "table_comment": table_comment})
                tables[table]["columns"].add(colname)
            desc = d.get("text","")
            tables[table]["comments"].append(desc)
            tables[table]["doc_ids"].append(meta.get("doc_id",""))
            
        elif meta.get("doc_type") == "table_comment" and table:
            tables.setdefault(table, {"columns": set(), "comments": [], "doc_ids": [], "table_comment": table_comment})
            tables[table]["comments"].append(d.get("text",""))
            tables[table]["doc_ids"].append(meta.get("chunk_index",""))
    
    # convert sets -> sorted lists
    for t in list(tables.keys()):
        tables[t]["columns"] = sorted(list(tables[t]["columns"]))
    
    # SMART ENHANCEMENT: Use semantic search to get only relevant columns
    # This reduces prompt size while maintaining accuracy
    if collection and original_query:
        for table_name in list(tables.keys()):
            try:
                # Use the passed original query for semantic column filtering
                
                # Query ChromaDB for semantically relevant columns of this specific table
                # Use the original query + table context for better column relevance
                column_query = f"{original_query} columns in table {table_name}"
                relevant_columns_results = collection.query(
                    query_texts=[column_query],
                    n_results=200,  # Get more columns for better coverage
                    where={"table": table_name.upper()}
                )
                
                if relevant_columns_results and relevant_columns_results['documents']:
                    for doc_text in relevant_columns_results['documents'][0]:
                        # Extract column information from the document
                        if "COLUMN:" in doc_text and f"TABLE: {table_name.upper()}" in doc_text:
                            # Parse column name from the document
                            lines = doc_text.split('\n')
                            for line in lines:
                                if line.strip().startswith("COLUMN:"):
                                    col_name = line.split("COLUMN:")[1].strip()
                                    if col_name and not is_generic_column(col_name):
                                        tables[table_name]["columns"].append(col_name)
                                        tables[table_name]["comments"].append(doc_text)
                                        break
                
                # Remove duplicates and sort
                tables[table_name]["columns"] = sorted(list(set(tables[table_name]["columns"])))
                
                # If we still don't have enough columns, get a few more essential ones
                
            except Exception as e:
                logger.warning(f"Could not retrieve relevant columns for table {table_name}: {e}")
    
    elif os.getenv("PINECONE_API_KEY") and original_query:
        # PINECONE MODE: Use targeted queries like ChromaDB for comprehensive column coverage
        logger.info("Pinecone mode: Using targeted column queries for comprehensive coverage")
        for table_name in list(tables.keys()):
            try:
                # Import Pinecone retrieval function
                from sqlgen_pinecone import retrieve_docs_semantic_pinecone
                
                # Query Pinecone for semantically relevant columns of this specific table
                # Use the original query + table context for better column relevance (same as ChromaDB)
                column_query = f"{original_query} columns in table {table_name}"
                column_results = retrieve_docs_semantic_pinecone(column_query, k=200)  # Same as ChromaDB
                
                if column_results.get("success") and column_results.get("docs"):
                    for d in column_results["docs"]:
                        doc_text = d.get("text", "")
                        meta = d.get("meta", {})
                        
                        # Check if this document is about the current table and is a column document
                        if (meta.get("table", "").upper() == table_name.upper() and 
                            meta.get("doc_type") == "column" and 
                            "COLUMN:" in doc_text):
                            
                            # Extract column name from the document
                            lines = doc_text.split('\n')
                            for line in lines:
                                if line.strip().startswith("COLUMN:"):
                                    col_name = line.split("COLUMN:")[1].strip()
                                    if col_name and not is_generic_column(col_name):
                                        tables[table_name]["columns"].append(col_name)
                                        tables[table_name]["comments"].append(doc_text)
                                        break
                
                # Remove duplicates and sort
                tables[table_name]["columns"] = sorted(list(set(tables[table_name]["columns"])))
                logger.info(f"Pinecone mode: Found {len(tables[table_name]['columns'])} columns for table {table_name}")
                
                # If we still don't have enough columns, get essential ones (same as ChromaDB)
                if len(tables[table_name]["columns"]) < 5:
                    essential_columns_query = f"primary key foreign key essential columns {table_name}"
                    essential_results = retrieve_docs_semantic_pinecone(essential_columns_query, k=100)
                    
                    if essential_results.get("success") and essential_results.get("docs"):
                        for d in essential_results["docs"]:
                            doc_text = d.get("text", "")
                            meta = d.get("meta", {})
                            
                            if (meta.get("table", "").upper() == table_name.upper() and 
                                meta.get("doc_type") == "column" and 
                                "COLUMN:" in doc_text):
                                lines = doc_text.split('\n')
                                for line in lines:
                                    if line.strip().startswith("COLUMN:"):
                                        col_name = line.split("COLUMN:")[1].strip()
                                        if col_name and not is_generic_column(col_name):
                                            if col_name not in tables[table_name]["columns"]:
                                                tables[table_name]["columns"].append(col_name)
                                                tables[table_name]["comments"].append(doc_text)
                                            break
                
                # Final cleanup
                tables[table_name]["columns"] = sorted(list(set(tables[table_name]["columns"])))
                
            except Exception as e:
                logger.warning(f"Pinecone mode: Error retrieving columns for table {table_name}: {e}")
    
    # FK-AWARE TABLE INCLUSION for Pinecone mode
    if os.getenv("PINECONE_API_KEY"):
        tables = include_referenced_tables_from_fks_pinecone(tables, original_query)
    
    # Final cleanup: Remove duplicates and sort for all tables
    for table_name in list(tables.keys()):
        tables[table_name]["columns"] = sorted(list(set(tables[table_name]["columns"])))
        logger.info(f"Final table {table_name}: {len(tables[table_name]['columns'])} columns")
    
    logger.info(f"Found {len(tables)} relevant tables after filtering")
    for table_name, table_data in tables.items():
        logger.debug(f"Table: {table_name}, Priority: {table_data.get('priority_score', 0):.1f}, Columns: {len(table_data['columns'])}, Comments: {len(table_data['comments'])}")
    
    return tables

def include_referenced_tables_from_fks_pinecone(tables: Dict[str, Dict], original_query: str = "") -> Dict[str, Dict]:
    """
    Pinecone-compatible version of FK relationship inclusion.
    Uses Pinecone queries to find FK relationships and include missing referenced tables.
    """
    logger.info("Analyzing FK relationships to include missing referenced tables (Pinecone mode)")
    
    try:
        from sqlgen_pinecone import retrieve_docs_semantic_pinecone
        
        # Collect all referenced tables from FK relationships
        referenced_tables = set()
        
        for table_name, table_data in tables.items():
            foreign_keys = table_data.get("foreign_keys", [])
            for fk in foreign_keys:
                referenced_table = fk.get("references_table")
                if referenced_table and referenced_table not in tables:
                    referenced_tables.add(referenced_table)
                    logger.info(f"Found FK reference to missing table: {table_name} -> {referenced_table}")
        
        # If no referenced tables are missing, return early
        if not referenced_tables:
            logger.info("All FK-referenced tables are already included")
            return tables
        
        logger.info(f"Including {len(referenced_tables)} missing referenced tables: {list(referenced_tables)}")
        
        # Fetch the missing referenced tables from Pinecone
        for referenced_table in referenced_tables:
            try:
                # Query Pinecone for table document
                table_query = f"table {referenced_table} primary key foreign key"
                table_results = retrieve_docs_semantic_pinecone(table_query, k=50)
                
                if not table_results.get("success") or not table_results.get("docs"):
                    logger.warning(f"Referenced table {referenced_table} not found in Pinecone")
                    continue
                
                # Find the table document
                table_doc = None
                table_meta = None
                for d in table_results["docs"]:
                    meta = d.get("meta", {})
                    if (meta.get("table", "").upper() == referenced_table.upper() and 
                        meta.get("doc_type") == "table"):
                        table_doc = d.get("text", "")
                        table_meta = meta
                        break
                
                if not table_doc:
                    logger.warning(f"No table document found for {referenced_table}")
                    continue
                
                # Check if this is a non-transactional table (skip if so)
                full_comment = get_full_table_comment(referenced_table, table_doc)
                if is_non_transactional_table(full_comment):
                    logger.info(f"Skipping non-transactional referenced table: {referenced_table}")
                    continue
                
                # Extract table information
                primary_key = extract_primary_key([table_doc])
                foreign_keys = extract_foreign_keys([table_doc])
                
                # Extract columns from table document
                columns = set()
                for line in table_doc.split('\n'):
                    if line.strip().upper().startswith('COLUMNS:'):
                        cols = line.split(':', 1)[1].strip()
                        columns = {c.strip() for c in cols.split(',') if c.strip() and not is_generic_column(c.strip())}
                        break
                
                # Get additional column information from Pinecone
                try:
                    column_query = f"essential columns {referenced_table}"
                    column_results = retrieve_docs_semantic_pinecone(column_query, k=100)
                    
                    if column_results.get("success") and column_results.get("docs"):
                        for d in column_results["docs"]:
                            doc_text = d.get("text", "")
                            meta = d.get("meta", {})
                            
                            if (meta.get("table", "").upper() == referenced_table.upper() and 
                                meta.get("doc_type") == "column" and 
                                "COLUMN:" in doc_text):
                                lines = doc_text.split('\n')
                                for line in lines:
                                    if line.strip().startswith("COLUMN:"):
                                        col_name = line.split("COLUMN:")[1].strip()
                                        if col_name and not is_generic_column(col_name):
                                            columns.add(col_name)
                except Exception as e:
                    logger.warning(f"Could not retrieve additional columns for referenced table {referenced_table}: {e}")
                
                # Create table data structure
                table_data = {
                    "columns": sorted(list(columns)),
                    "comments": [table_doc],
                    "doc_ids": [table_meta.get("doc_id", "")],
                    "table_comment": table_meta.get("table_comment", ""),
                    "primary_key": primary_key,
                    "foreign_keys": foreign_keys,
                    "priority_score": 0.0  # Will be calculated later
                }
                
                # Calculate priority score for the referenced table
                table_info = {"comments": [table_doc], "columns": list(columns)}
                table_data["priority_score"] = calculate_table_priority_score(referenced_table, table_info, original_query)
                
                # Add to tables dictionary
                tables[referenced_table] = table_data
                logger.info(f"Added referenced table {referenced_table} with {len(columns)} columns, priority: {table_data['priority_score']:.1f}")
                
            except Exception as e:
                logger.warning(f"Could not include referenced table {referenced_table}: {e}")
                continue
        
        logger.info(f"FK-aware table inclusion complete. Total tables: {len(tables)}")
        return tables
        
    except Exception as e:
        logger.warning(f"FK-aware table inclusion failed: {e}")
        return tables

def include_referenced_tables_from_fks(tables: Dict[str, Dict], collection, original_query: str = "") -> Dict[str, Dict]:
    """
    Dynamically include referenced tables from FK relationships to ensure complete JOIN paths.
    This function analyzes FK relationships in the selected tables and automatically includes
    any referenced tables that are missing, ensuring the LLM has all necessary tables for valid JOINs.
    """
    logger.info("Analyzing FK relationships to include missing referenced tables")
    
    # Collect all referenced tables from FK relationships
    referenced_tables = set()
    
    for table_name, table_data in tables.items():
        foreign_keys = table_data.get("foreign_keys", [])
        for fk in foreign_keys:
            referenced_table = fk.get("references_table")
            if referenced_table and referenced_table not in tables:
                referenced_tables.add(referenced_table)
                logger.info(f"Found FK reference to missing table: {table_name} -> {referenced_table}")
    
    # If no referenced tables are missing, return early
    if not referenced_tables:
        logger.info("All FK-referenced tables are already included")
        return tables
    
    logger.info(f"Including {len(referenced_tables)} missing referenced tables: {list(referenced_tables)}")
    
    # Fetch the missing referenced tables from ChromaDB
    for referenced_table in referenced_tables:
        try:
            # Get table document
            table_results = collection.get(
                where={"$and": [{"table": referenced_table.upper()}, {"doc_type": "table"}]},
                limit=1
            )
            
            if not table_results['documents']:
                logger.warning(f"Referenced table {referenced_table} not found in ChromaDB")
                continue
            
            table_doc = table_results['documents'][0]
            table_meta = table_results['metadatas'][0]
            
            # Check if this is a non-transactional table (skip if so)
            full_comment = get_full_table_comment(referenced_table, table_doc)
            if not full_comment:
                full_comment = table_meta.get("table_comment", "")
            
            if is_non_transactional_table(full_comment):
                logger.info(f"Skipping non-transactional referenced table: {referenced_table}")
                continue
            
            # Extract table information
            primary_key = extract_primary_key([table_doc])
            foreign_keys = extract_foreign_keys([table_doc])
            
            # If no FK relationships found, this is already a doc_type: 'table' document
            # so the FK relationships should be in the document content
            
            # Extract columns from table document
            columns = set()
            for line in table_doc.split('\n'):
                if line.strip().upper().startswith('COLUMNS:'):
                    cols = line.split(':', 1)[1].strip()
                    columns = {c.strip() for c in cols.split(',') if c.strip() and not is_generic_column(c.strip())}
                    break
            
            # Get additional column information from ChromaDB
            try:
                column_results = collection.query(
                    query_texts=[f"essential columns {referenced_table}"],
                    n_results=50,
                    where={"$and": [{"table": referenced_table.upper()}, {"doc_type": "column"}]}
                )
                
                if column_results and column_results['documents']:
                    for doc_text in column_results['documents'][0]:
                        if "COLUMN:" in doc_text and f"TABLE: {referenced_table.upper()}" in doc_text:
                            lines = doc_text.split('\n')
                            for line in lines:
                                if line.strip().startswith("COLUMN:"):
                                    col_name = line.split("COLUMN:")[1].strip()
                                    if col_name and not is_generic_column(col_name):
                                        columns.add(col_name)
            except Exception as e:
                logger.warning(f"Could not retrieve additional columns for referenced table {referenced_table}: {e}")
            
            # Create table data structure
            table_data = {
                "columns": sorted(list(columns)),
                "comments": [table_doc],
                "doc_ids": [table_meta.get("doc_id", "")],
                "table_comment": table_meta.get("table_comment", ""),
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "priority_score": 0.0  # Will be calculated later
            }
            
            # Calculate priority score for the referenced table
            table_info = {"comments": [table_doc], "columns": list(columns)}
            table_data["priority_score"] = calculate_table_priority_score(referenced_table, table_info, original_query)
            
            # Add to tables dictionary
            tables[referenced_table] = table_data
            logger.info(f"Added referenced table {referenced_table} with {len(columns)} columns, priority: {table_data['priority_score']:.1f}")
            
        except Exception as e:
            logger.warning(f"Could not include referenced table {referenced_table}: {e}")
            continue
    
    logger.info(f"FK-aware table inclusion complete. Total tables: {len(tables)}")
    return tables

def analyze_query_requirements(user_query: str) -> Dict[str, int]:
    """
    Analyze the user query to determine what types of columns are needed.
    Returns a dictionary with column type requirements and their importance scores.
    """
    query_lower = user_query.lower()
    requirements = {
        'join_columns': 0,      # INVOICE_ID, CUSTOMER_ID, etc.
        'filter_columns': 0,    # STATUS, FLAG, DATE columns for WHERE clauses
        'display_columns': 0,   # NAME, DESCRIPTION, AMOUNT columns for SELECT
        'date_columns': 0,      # DATE columns for time-based queries
        'amount_columns': 0,    # AMOUNT, PRICE, COST columns for financial queries
        'status_columns': 0,    # STATUS, FLAG columns for state-based queries
    }
    
    # GENERALIZED SEMANTIC ANALYSIS (NO HARDCODING)
    try:
        # Define semantic patterns for different column types
        semantic_patterns = {
            'join_columns': [
                'join', 'with', 'related', 'associated', 'connect', 'link', 'between',
                'relationship', 'belong', 'own', 'reference', 'foreign', 'and'
            ],
            'filter_columns': [
                'where', 'filter', 'condition', 'criteria', 'unpaid', 'paid', 'status',
                'due', 'overdue', 'pending', 'active', 'inactive', 'valid', 'invalid'
            ],
            'display_columns': [
                'list', 'show', 'display', 'report', 'name', 'description', 'title',
                'label', 'text', 'detail', 'information', 'data', 'get', 'find'
            ],
            'date_columns': [
                'date', 'time', 'when', 'due', 'schedule', 'period', 'month', 'year',
                'august', 'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'september', 'october', 'november', 'december', 'created',
                'updated', 'modified', 'expired', 'valid', 'day'
            ],
            'amount_columns': [
                'amount', 'total', 'value', 'cost', 'price', 'sum', 'balance',
                'quantity', 'count', 'number', 'figure', 'money', 'currency',
                'dollar', 'euro', 'payment', 'receipt', 'invoice', 'financial'
            ],
            'status_columns': [
                'status', 'flag', 'type', 'state', 'condition', 'indicator',
                'marker', 'sign', 'signal', 'code', 'category', 'class'
            ]
        }
        
        # Calculate semantic relevance for each column type
        for col_type, patterns in semantic_patterns.items():
            relevance_score = 0
            for pattern in patterns:
                if pattern in query_lower:
                    # Weight by pattern importance and frequency
                    pattern_weight = 1.0
                    if pattern in ['join', 'where', 'list', 'date', 'amount', 'status']:
                        pattern_weight = 2.0  # Core SQL concepts
                    elif pattern in ['unpaid', 'due', 'invoice', 'payment']:
                        pattern_weight = 1.5  # Domain-specific terms
                    
                    relevance_score += pattern_weight
            
            # Normalize and scale the score
            requirements[col_type] = min(int(relevance_score * 3), 20)
            
    except Exception as e:
        logger.debug(f"Semantic analysis failed: {e}")
        # Fallback to simple keyword matching
        if any(word in query_lower for word in ['join', 'with', 'related']):
            requirements['join_columns'] = 15
        if any(word in query_lower for word in ['where', 'filter', 'due', 'status']):
            requirements['filter_columns'] = 12
        if any(word in query_lower for word in ['list', 'show', 'display']):
            requirements['display_columns'] = 10
        if any(word in query_lower for word in ['date', 'time', 'when']):
            requirements['date_columns'] = 15
        if any(word in query_lower for word in ['amount', 'total', 'value']):
            requirements['amount_columns'] = 12
        if any(word in query_lower for word in ['status', 'flag', 'type']):
            requirements['status_columns'] = 15
    
    return requirements

def determine_optimal_column_count(user_query: str, query_requirements: Dict[str, int]) -> int:
    """
    Determine the optimal number of columns to include based on query complexity.
    """
    query_lower = user_query.lower()
    
    # Base count
    base_columns = 3
    
    # Add columns based on query complexity
    if any(word in query_lower for word in ['join', 'with', 'and', 'related']):
        base_columns += 2  # Need more columns for JOINs
    
    if any(word in query_lower for word in ['list', 'show', 'display', 'report']):
        base_columns += 2  # Need more columns for display
    
    if any(word in query_lower for word in ['where', 'filter', 'due', 'status', 'amount']):
        base_columns += 1  # Need filter columns
    
    if any(word in query_lower for word in ['complex', 'detailed', 'comprehensive', 'full']):
        base_columns += 3  # Complex queries need more columns
    
    # Add columns based on requirements
    if query_requirements['join_columns'] > 10:
        base_columns += 1
    if query_requirements['display_columns'] > 10:
        base_columns += 2
    if query_requirements['filter_columns'] > 10:
        base_columns += 1
    
    # Ensure minimum and maximum bounds - increased to ensure we have enough columns
    min_columns = 5
    max_columns = 20  # Increased from 12 to 20 to ensure we have enough columns
    
    return max(min_columns, min(base_columns, max_columns))

def select_relevant_columns(table_name: str, table_data: Dict, user_query: str) -> List[tuple]:
    """
    Intelligently select the most relevant columns for a table based on:
    1. Query requirements analysis (DYNAMIC)
    2. Semantic relevance to the user query
    3. Column importance (IDs, dates, amounts, etc.)
    4. Column comment quality
    """
    query_requirements = analyze_query_requirements(user_query)
    query_terms = set(user_query.lower().split())
    relevant_columns = []
    
    for col in table_data.get("columns", []):
        # Find the comment for this specific column in this specific table
        col_comment = ""
        for comment in table_data.get("comments", []):
            if f"COLUMN: {col}" in comment and f"TABLE: {table_name}" in comment:
                col_comment = comment
                break
        
        # Extract semantic meaning from column comment
        col_semantic = col_comment if col_comment else f"Column: {col}"
        
        # Calculate relevance score
        relevance_score = 0
        
        # 1. Query requirements matching (DYNAMIC)
        col_upper = col.upper()
        
        # Join columns (IDs)
        if any(keyword in col_upper for keyword in ['_ID', '_NUMBER', '_CODE']):
            relevance_score += query_requirements['join_columns']
        
        # Filter columns (STATUS, FLAG, DATE)
        if any(keyword in col_upper for keyword in ['_STATUS', '_FLAG', '_TYPE']):
            relevance_score += query_requirements['status_columns']
        
        # Date columns
        if any(keyword in col_upper for keyword in ['_DATE', '_DT', '_TIME']):
            relevance_score += query_requirements['date_columns']
        
        # Amount columns
        if any(keyword in col_upper for keyword in ['_AMOUNT', '_AMT', '_VALUE', '_COST', '_PRICE', '_TOTAL']):
            relevance_score += query_requirements['amount_columns']
        
        # Display columns (NAME, DESCRIPTION)
        if any(keyword in col_upper for keyword in ['_NAME', '_DESCRIPTION', '_COMMENT', '_TITLE']):
            relevance_score += query_requirements['display_columns']
        
        # 2. SEMANTIC MEANING MATCHING using column comments (HIGH PRIORITY)
        if col_comment:
            col_comment_lower = col_comment.lower()
            
            # Extract semantic meaning from column comment
            col_semantic = col_comment
            
            # Match query intent with column semantic meaning
            query_lower = user_query.lower()
            
            # GENERALIZED SEMANTIC MATCHING - No hardcoded terms
            # Use the query requirements analysis to determine relevance
            query_requirements = analyze_query_requirements(user_query)
            
            # Match column comment with query requirements
            for req_type, req_score in query_requirements.items():
                if req_score > 0:  # Only if this requirement is relevant to the query
                    # Define semantic keywords for each requirement type
                    semantic_keywords = {
                        'join_columns': ['identifier', 'reference', 'foreign key', 'relationship', 'link', 'associate'],
                        'filter_columns': ['status', 'flag', 'condition', 'criteria', 'indicator', 'marker'],
                        'display_columns': ['name', 'description', 'title', 'label', 'text', 'detail'],
                        'date_columns': ['date', 'time', 'when', 'schedule', 'period', 'created', 'updated'],
                        'amount_columns': ['amount', 'total', 'value', 'cost', 'price', 'sum', 'balance'],
                        'status_columns': ['status', 'flag', 'type', 'state', 'condition', 'indicator']
                    }
                    
                    # Check if column comment matches this requirement type
                    if any(keyword in col_comment_lower for keyword in semantic_keywords.get(req_type, [])):
                        relevance_score += req_score * 0.5  # Scale by requirement importance
            
            # General semantic matching - look for conceptual similarity
            for term in query_terms:
                if len(term) > 3:  # Only match meaningful terms
                    if term in col_comment_lower:
                        relevance_score += 6  # Good semantic match
                    elif any(word in col_comment_lower for word in [term + 's', term + 'ed', term + 'ing']):
                        relevance_score += 4  # Partial semantic match
        
        # 4. GENERALIZED SEMANTIC RELEVANCE MATCHING (NO HARDCODING)
        if col_comment:
            # Use the embedding model to calculate semantic similarity
            try:
                # Get embeddings for query and column comment
                model = get_global_embed_model()
                query_embedding = model.encode([user_query])
                col_embedding = model.encode([col_comment])
                
                # Calculate cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(query_embedding, col_embedding)[0][0]
                
                # Convert similarity to relevance score (0-1 range to 0-20 range)
                semantic_relevance = similarity * 20
                relevance_score += semantic_relevance
                
            except Exception as e:
                # Fallback to simple text similarity if embedding fails
                logger.debug(f"Embedding similarity failed for {col}: {e}")
                # Simple fallback: count common meaningful words
                query_words = set(user_query.lower().split())
                comment_words = set(col_comment.lower().split())
                common_words = query_words.intersection(comment_words)
                if common_words:
                    relevance_score += len(common_words) * 2
        
        # 5. Primary key bonus
        if col in table_data.get("primary_key", []):
            relevance_score += 10  # Primary keys are CRITICAL for JOINs
        
        # 6. Foreign key bonus
        if table_data.get("foreign_keys"):
            for fk in table_data["foreign_keys"]:
                if col == fk["source_column"]:
                    relevance_score += 2
                    break
        
        # Only include columns with meaningful relevance or essential columns
        if relevance_score > 0 or any(keyword in col_upper for keyword in ['_ID', '_NUMBER', '_DATE', '_AMOUNT']):
            relevant_columns.append((col, col_semantic, relevance_score))
    
    # Sort by relevance score (highest first)
    relevant_columns.sort(key=lambda x: x[2], reverse=True)
    
    # DYNAMIC column selection based on query complexity
    max_columns = determine_optimal_column_count(user_query, query_requirements)
    
    # CRITICAL: Ensure we always include essential JOIN columns even if they don't score high
    essential_join_columns = []
    for col in table_data.get("columns", []):
        col_upper = col.upper()
        col_desc = ""
        
        # Find the comment for this specific column
        for comment in table_data.get("comments", []):
            if f"COLUMN: {col}" in comment and f"TABLE: {table_name}" in comment:
                col_desc = comment
                break
        
        # Always include primary key columns and common JOIN columns
        is_essential_join = False
        
        # Check if it's a primary key
        if "primary key" in col_desc.lower():
            is_essential_join = True
        
        # Check if it's an identifier column
        if "identifier" in col_desc.lower() and col_upper.endswith("_ID"):
            is_essential_join = True
        
        # Check if it's a common JOIN column (GENERALIZED PATTERN)
        if col_upper.endswith("_ID"):
            # Check if the column comment indicates it's a relationship/identifier
            if any(word in col_desc.lower() for word in [
                'identifier', 'reference', 'foreign key', 'relationship', 'link', 
                'associate', 'connect', 'relate', 'belong', 'own'
            ]):
                is_essential_join = True
        
        # Check if it's referenced in foreign key relationships
        if table_data.get("foreign_keys"):
            for fk in table_data["foreign_keys"]:
                if col == fk["source_column"]:
                    is_essential_join = True
                    break
        
        if is_essential_join and not any(existing[0] == col for existing in relevant_columns[:max_columns]):
            essential_join_columns.append((col, col_desc, 15.0))  # Very high score for essential JOIN columns
    
    # Add essential JOIN columns if not already included
    final_columns = relevant_columns[:max_columns]
    for essential_col in essential_join_columns:
        if not any(existing[0] == essential_col[0] for existing in final_columns):
            final_columns.append(essential_col)
    
    return final_columns

def add_join_suggestions(tables: Dict[str, Dict], collection, original_query: str):
    """
    Add JOIN suggestions for missing columns to help the LLM create better queries.
    Uses semantic search to find relevant columns based on the user query.
    """
    logger.info("Adding JOIN suggestions for missing columns using semantic search")
    
    # Use semantic search to find columns relevant to the user query
    query_emb = embed_query(original_query)
    
    # Find missing columns in high priority tables
    high_priority_tables = [(t, data) for t, data in tables.items() if data.get('priority_score', 0) > 5]
    
    for table_name, table_data in high_priority_tables:
        # Search for columns semantically related to the user query
        semantic_column_query = f"{original_query} columns"
        results = collection.query(
            query_texts=[semantic_column_query],
            n_results=50,
            where={"doc_type": "column"}
        )
        
        if results and results['documents']:
            for doc_text in results['documents'][0]:
                if "COLUMN:" in doc_text:
                    # Extract column name from the document
                    for line in doc_text.split('\n'):
                        if line.strip().startswith("COLUMN:"):
                            col_name = line.split("COLUMN:")[1].strip()
                            # Extract table name from the document
                            for table_line in doc_text.split('\n'):
                                if table_line.strip().startswith("TABLE:"):
                                    source_table = table_line.split("TABLE:")[1].strip()
                                    
                                    # If this column is not in the current table but exists in another table
                                    if (col_name not in table_data.get('columns', []) and 
                                        source_table != table_name and 
                                        source_table in tables):
                                        
                                        # Add JOIN suggestion
                                        if 'join_suggestions' not in table_data:
                                            table_data['join_suggestions'] = []
                                        table_data['join_suggestions'].append({
                                            'column': col_name,
                                            'source_table': source_table,
                                            'reason': f"Table {table_name} missing {col_name}, available in {source_table}"
                                        })
                                        logger.info(f"JOIN suggestion: {table_name} -> {source_table} for {col_name}")
                                    break
                            break

# ---------- Enhanced Prompt ----------
SYSTEM_PROMPT = """Generate Oracle SQL query using the provided schema.

CRITICAL RULES:
1. ALWAYS generate valid SQL - NEVER return error messages or comments about missing columns
2. Use ONLY columns that are explicitly listed in the ALL_AVAILABLE_COLUMNS section
3. Use table aliases: a, b, c
4. For dates: TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')
5. Copy column names exactly as shown in the schema
6. Understand business logic from column descriptions and table purposes
7. Use appropriate SQL functions based on query requirements

8. CRITICAL: Column Selection Process:
   - FIRST: Check ALL_AVAILABLE_COLUMNS to see what columns actually exist
   - SECOND: Use column descriptions to understand what each column represents
   - THIRD: Map your query requirements to existing columns
   - NEVER assume column names - always verify they exist in ALL_AVAILABLE_COLUMNS

9. CRITICAL: For calculated values like "unpaid amount":
   - Analyze column descriptions to understand business logic
   - Look for columns that represent "paid amounts" vs "total amounts" 
   - Calculate derived values using existing columns (e.g., total - paid = unpaid)
   - Use SUM() with GROUP BY for aggregations
   - NEVER use aliases as column names - always calculate from actual columns
   - If you need a calculated value, examine the column descriptions to find the right columns to use
   - THINK STEP BY STEP: What columns represent the total amount? What columns represent paid amounts? Then calculate the difference

10. CRITICAL: Data type awareness:
    - ID columns (ending with _ID) are typically NUMBER type - compare with numbers, not dates
    - DATE columns are DATE type - compare with TO_DATE() functions
    - Never compare NUMBER columns with DATE functions
    - For date filtering on ID columns, use the actual date column from the related table
    - Example: If filtering by "created before 2024", use the date column, not the ID column

11. CRITICAL: If a column doesn't exist:
    - Find the semantically equivalent column from ALL_AVAILABLE_COLUMNS
    - Use the column description to understand what it represents
    - NEVER return error messages - always generate valid SQL using existing columns

MANDATORY OUTPUT FORMAT:
SQL:
[Your valid Oracle SQL query using only existing columns]"""

def find_semantically_related_columns(primary_col: str, primary_desc: str, all_columns: List, model, added_columns: set) -> List[Tuple[str, str, str]]:
    """
    Find semantically related columns using embeddings without hardcoded keywords.
    This is truly dynamic and works for any type of query.
    """
    import numpy as np
    
    if not primary_desc.strip():
        return []
    
    # Create embedding for the primary column description
    primary_embedding = model.encode([primary_desc])
    
    related_columns = []
    
    # Find columns with high semantic similarity to the primary column
    for (col_name, col_desc, col_type, col_comment) in all_columns:
        if (col_name not in added_columns and 
            col_name != primary_col and 
            col_desc.strip()):
            
            # Calculate semantic similarity between column descriptions
            col_embedding = model.encode([col_desc])
            similarity = np.dot(primary_embedding, col_embedding.T).flatten()[0]
            
            # If highly semantically related, include it
            if similarity > 0.6:  # High semantic similarity threshold
                related_columns.append((col_name, col_desc, col_type))
                
                # Limit to avoid overwhelming the LLM
                if len(related_columns) >= 3:
                    break
    
    return related_columns

def get_essential_columns_for_table(table_name: str, table_data: Dict[str, Any], user_query: str) -> List[Tuple[str, str, str]]:
    """
    Get essential columns for a table based on semantic similarity to the user query.
    Uses column comments to find relevant columns while excluding generic patterns.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Load the embedding model
    try:
        model = get_embed_model()
    except Exception as e:
        logger.warning(f"Could not load embedding model for column selection: {e}")
        # Fallback to simple keyword matching
        return get_essential_columns_fallback(table_name, table_data, user_query)
    
    # Get all columns with their comments
    all_columns = []
    for col in table_data.get("columns", []):
        # Find the comment for this specific column
        col_comment = ""
        for comment in table_data.get("comments", []):
            if f"COLUMN: {col}" in comment and f"TABLE: {table_name}" in comment:
                col_comment = comment
                break
        
        # Extract data type from column description
        col_type = "UNKNOWN"
        if col_comment:
            for line in col_comment.split('\n'):
                if line.strip().startswith("TYPE:"):
                    col_type = line.split("TYPE:")[1].strip()
                    break
        
        # Extract description from comment
        col_desc = ""
        if col_comment:
            for line in col_comment.split('\n'):
                if line.strip().startswith("DESCRIPTION:"):
                    col_desc = line.split("DESCRIPTION:")[1].strip()
                    break
        
        all_columns.append((col, col_desc, col_type, col_comment))
    
    # Filter out generic columns
    filtered_columns = []
    for col_name, col_desc, col_type, col_comment in all_columns:
        # Check if column matches generic patterns
        is_generic = False
        for pattern in GENERIC_COLUMN_PATTERNS:
            if pattern.lower() in col_name.lower():
                is_generic = True
                break
        
        if not is_generic:
            filtered_columns.append((col_name, col_desc, col_type, col_comment))
    
    if not filtered_columns:
        return []
    
    # Always include primary key columns
    essential_columns = []
    for col_name, col_desc, col_type, col_comment in filtered_columns:
        if "PRIMARY_KEY: YES" in col_comment:
            essential_columns.append((col_name, col_desc, col_type))
    
    # Calculate semantic similarity between user query and column descriptions
    query_embedding = model.encode([user_query])
    column_descriptions = [f"{col_name}: {col_desc}" for col_name, col_desc, col_type, col_comment in filtered_columns]
    column_embeddings = model.encode(column_descriptions)
    
    # Calculate cosine similarity
    similarities = np.dot(query_embedding, column_embeddings.T).flatten()
    
    # Sort columns by similarity score
    scored_columns = list(zip(filtered_columns, similarities))
    scored_columns.sort(key=lambda x: x[1], reverse=True)
    
    # Add top relevant columns (excluding those already added as primary keys)
    added_columns = {col[0] for col in essential_columns}
    # Add columns based on semantic similarity
    for (col_name, col_desc, col_type, col_comment), similarity in scored_columns:
        if col_name not in added_columns and similarity > 0.3:
            essential_columns.append((col_name, col_desc, col_type))
            added_columns.add(col_name)
            
            # Find semantically related columns using embeddings
            if similarity > 0.4:  # Only for highly relevant columns
                related_columns = find_semantically_related_columns(
                    col_name, col_desc, filtered_columns, model, added_columns
                )
                for related_col, related_desc, related_type in related_columns:
                    if len(essential_columns) < 15:
                        essential_columns.append((related_col, related_desc, related_type))
                        added_columns.add(related_col)
    
    # Limit to maximum 15 columns per table to avoid overwhelming the LLM
    if len(essential_columns) > 15:
        essential_columns = essential_columns[:15]
    
    logger.debug(f"Selected {len(essential_columns)} essential columns for {table_name}")
    return essential_columns

def get_essential_columns_fallback(table_name: str, table_data: Dict[str, Any], user_query: str) -> List[Tuple[str, str, str]]:
    """
    Fallback method for column selection using simple keyword matching.
    """
    essential_columns = []
    query_lower = user_query.lower()
    
    for col in table_data.get("columns", []):
        # Check if column matches generic patterns
        is_generic = False
        for pattern in GENERIC_COLUMN_PATTERNS:
            if pattern.lower() in col.lower():
                is_generic = True
                break
        
        if is_generic:
            continue
        
        # Find the comment for this specific column
        col_comment = ""
        for comment in table_data.get("comments", []):
            if f"COLUMN: {col}" in comment and f"TABLE: {table_name}" in comment:
                col_comment = comment
                break
        
        # Extract data type and description
        col_type = "UNKNOWN"
        col_desc = ""
        if col_comment:
            for line in col_comment.split('\n'):
                if line.strip().startswith("TYPE:"):
                    col_type = line.split("TYPE:")[1].strip()
                elif line.strip().startswith("DESCRIPTION:"):
                    col_desc = line.split("DESCRIPTION:")[1].strip()
        
        # Always include primary key columns
        if "PRIMARY_KEY: YES" in col_comment:
            essential_columns.append((col, col_desc, col_type))
        # Include ID columns for JOINs
        elif col.endswith('_ID'):
            essential_columns.append((col, col_desc, col_type))
        # Include columns based on query context
        elif any(word in query_lower for word in ['amount', 'total', 'sum', 'cost', 'price']) and any(word in col.lower() for word in ['amount', 'total', 'cost', 'price', 'value']):
            essential_columns.append((col, col_desc, col_type))
        elif any(word in query_lower for word in ['date', 'time', 'created', 'updated', 'due']) and any(word in col.lower() for word in ['date', 'time', 'created', 'updated', 'due']):
            essential_columns.append((col, col_desc, col_type))
        elif any(word in query_lower for word in ['status', 'flag', 'state']) and any(word in col.lower() for word in ['status', 'flag', 'state']):
            essential_columns.append((col, col_desc, col_type))
        elif any(word in query_lower for word in ['name', 'description', 'title']) and any(word in col.lower() for word in ['name', 'description', 'title', 'num', 'number']):
            essential_columns.append((col, col_desc, col_type))
    
    # Limit to maximum 15 columns per table
    if len(essential_columns) > 15:
        essential_columns = essential_columns[:15]
    
    return essential_columns

def get_process_context_for_query(user_query: str) -> str:
    """
    Get relevant business process context for the user query using ChromaDB.
    This helps the LLM understand which business process the query relates to.
    Enhanced with caching for better performance.
    """
    global _process_context_cache
    
    # Check cache first
    query_key = user_query.lower().strip()
    if query_key in _process_context_cache:
        logger.debug("Using cached process context")
        return _process_context_cache[query_key]
    
    try:
        # Connect to ChromaDB
        client = get_chroma_client()
        collection = get_chroma_collection()
        
        # Load embedding model
        model = get_embed_model()
        
        # Create query embedding
        query_embedding = model.encode([user_query]).tolist()[0]
        
        # Search for relevant process documentation
        # Try both with and without document_type filter since metadata might vary
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"document_type": "business_process"}
            )
        except:
            # Fallback: search all documents and filter by content
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10
            )
            # Filter results to only include process documents
            if results['documents'] and results['documents'][0]:
                filtered_docs = []
                filtered_metadatas = []
                for i, doc in enumerate(results['documents'][0]):
                    if doc and ("PROCESS:" in doc or "OVERVIEW:" in doc or "STEPS:" in doc):
                        filtered_docs.append(doc)
                        if results['metadatas'] and results['metadatas'][0]:
                            filtered_metadatas.append(results['metadatas'][0][i])
                results = {
                    'documents': [filtered_docs[:3]],  # Limit to top 3
                    'metadatas': [filtered_metadatas[:3]] if filtered_metadatas else None
                }
        
        if results['documents'] and results['documents'][0]:
            process_context = []
            for i, doc in enumerate(results['documents'][0]):
                if doc and doc.strip():
                    process_context.append(f"PROCESS_CONTEXT_{i+1}: {doc}")
            
            if process_context:
                context_result = "\n".join(process_context)
                # Cache the result
                _process_context_cache[query_key] = context_result
                logger.info(f"Found {len(process_context)} relevant process contexts")
                return context_result
        
        logger.info("No relevant process context found")
        # Cache empty result to avoid repeated searches
        _process_context_cache[query_key] = ""
        return ""
        
    except Exception as e:
        logger.warning(f"Failed to get process context: {e}")
        # Cache empty result to avoid repeated failures
        _process_context_cache[query_key] = ""
        return ""

def filter_process_context_for_existing_tables(process_context: str, available_tables: List[str]) -> str:
    """
    Filter process context to only mention tables that actually exist in our schema.
    This prevents LLM hallucination of non-existent tables.
    """
    if not process_context or not available_tables:
        return ""
    
    # Convert to uppercase for comparison
    available_tables_upper = [t.upper() for t in available_tables]
    
    lines = []
    for line in process_context.split('\n'):
        # Check if line mentions tables
        if '- Tables:' in line or 'Tables:' in line:
            # Split the line and filter out non-existent tables
            parts = line.split(':')
            if len(parts) >= 2:
                table_list = parts[1].strip()
                # Filter out non-existent tables
                existing_tables = []
                for table in table_list.split(','):
                    table = table.strip().upper()
                    if table in available_tables_upper:
                        existing_tables.append(table)
                
                # Only include line if it has existing tables
                if existing_tables:
                    filtered_line = parts[0] + ': ' + ', '.join(existing_tables)
                    lines.append(filtered_line)
        else:
            # Include non-table lines as-is
            lines.append(line)
    
    return '\n'.join(lines)

def build_semantic_prompt(user_query: str, tables_summary: Dict[str, Any], available_tables: List[str] = None) -> str:
    """
    Build a simple, focused prompt to avoid overwhelming the LLM.
    """
    logger.info("Building simplified prompt for LLM")
    
    # Sort tables by priority score and take only top 2
    sorted_tables = sorted(tables_summary.items(), 
                          key=lambda x: x[1].get('priority_score', 0), 
                          reverse=True)
    
    lines = ["Schema with column-to-table mapping:"]
    
    # TEMPORARILY DISABLED: Process context causes LLM hallucination
    # Get process context for better business understanding (filtered to only existing tables)
    # process_context = get_process_context_for_query(user_query)
    # if process_context and available_tables:
    #     # Filter process context to only mention tables that actually exist
    #     filtered_context = filter_process_context_for_existing_tables(process_context, available_tables)
    #     if filtered_context:
    #         lines.append(f"\n{filtered_context}")
    #         lines.append("")  # Add spacing
    
    # CRITICAL: Add available tables list to prevent hallucination
    if available_tables:
        lines.append(f"AVAILABLE_TABLES: {', '.join(available_tables)}")
        lines.append("CRITICAL: ONLY use tables from the AVAILABLE_TABLES list above - NEVER invent table names!")
        lines.append("")  # Add spacing
    
    # Show top 3 tables with explicit column mapping
    top_tables = sorted_tables[:3]
    
    for t, data in top_tables:
        lines.append(f"\nTABLE: {t}")
        
        # Get essential columns
        essential_columns = get_essential_columns_for_table(t, data, user_query)
        
        if essential_columns:
            lines.append("COLUMNS:")
            for col_name, col_desc, col_type in essential_columns:
                # Include column description for semantic understanding
                if col_desc and col_desc.strip():
                    lines.append(f"  {t}.{col_name} ({col_type}) - {col_desc}")
                else:
                    lines.append(f"  {t}.{col_name} ({col_type})")
            
            # Add all available columns for this table to prevent assumptions
            all_columns = data.get("columns", [])
            if all_columns and len(all_columns) > len(essential_columns):
                lines.append(f"ALL_AVAILABLE_COLUMNS: {', '.join([f'{t}.{col}' for col in all_columns[:20]])}")  # Limit to first 20
                if len(all_columns) > 20:
                    lines.append(f"... and {len(all_columns) - 20} more columns")
        
        # Add table comment for context
        table_comment = data.get("table_comment", "")
        if table_comment and table_comment.strip():
            lines.append(f"TABLE_PURPOSE: {table_comment}")
        
        # Show key relationships
        if data.get("foreign_keys"):
            lines.append("FOREIGN KEYS:")
            for fk in data["foreign_keys"][:3]:
                lines.append(f"  {t}.{fk['source_column']} -> {fk['references_table']}.{fk['references_column']}")
    
    # Add dynamic guidance with explicit column mapping
    if top_tables:
        lines.append("\nCRITICAL RULES:")
        lines.append("- ALWAYS use table.column format (e.g., table_name.column_name)")
        lines.append("- Each column belongs to ONLY ONE table - check the mapping above")
        lines.append("- JOIN tables using foreign key relationships shown above")
        lines.append("- If you need a column, find which table it belongs to first")
        lines.append("- Use column descriptions to understand business logic and relationships")
        lines.append("- For aggregations, use appropriate SQL functions based on query intent")
        lines.append("- Analyze column descriptions to understand how to calculate derived values")
        lines.append("- CRITICAL: Only use tables from the AVAILABLE_TABLES list - NEVER invent table names!")
        lines.append("- CRITICAL: Only use columns that are explicitly listed in ALL_AVAILABLE_COLUMNS above")
        lines.append("- If you need a column that's not listed, check if there's a semantically similar column")
        lines.append("- Do NOT assume column names - verify they exist in the schema provided")
        lines.append("- ALWAYS generate valid SQL - NEVER return error messages about missing columns")
        lines.append("- If a column doesn't exist, find the semantically equivalent column from ALL_AVAILABLE_COLUMNS")
        lines.append("- NEVER invent or hallucinate column names - only use what's provided in ALL_AVAILABLE_COLUMNS")
        lines.append("- NEVER invent or hallucinate table names - only use what's in AVAILABLE_TABLES")
        lines.append("- If you need a specific type of column, search through ALL_AVAILABLE_COLUMNS to find the exact name")
        lines.append("- Double-check every column name against the ALL_AVAILABLE_COLUMNS list before using it")
        lines.append("- Double-check every table name against the AVAILABLE_TABLES list before using it")
        lines.append("- CRITICAL: Define ALL table aliases in FROM and JOIN clauses before using them")
        lines.append("- NEVER use undefined table aliases - always define them first in FROM/JOIN clauses")
        lines.append("- ALWAYS use the SAME alias you defined in FROM/JOIN - don't mix up aliases!")
        lines.append("- Example: If you define 'FROM AP_PAYMENT_SCHEDULES_ALL a', then use 'a.COLUMN_NAME', not 'b.COLUMN_NAME'")
        lines.append("- Use the table names and foreign key relationships shown above to create proper JOINs")
    
    lines.append(f"\nQuery: {user_query}")
    
    prompt = "\n".join(lines)
    logger.debug(f"Simplified prompt:\n{prompt}")
    return prompt

# ---------- Ollama REST call (via requests) ----------
def correct_sql_dynamically(sql: str, schema_columns: Dict[str, List[str]]) -> str:
    """
    Dynamically correct SQL errors by analyzing the schema and fixing common patterns.
    This is production-ready and works with any schema.
    """
    import re
    
    corrected_sql = sql
    
    # CRITICAL: Fix undefined table aliases
    # This is the most common issue - LLM generating SQL with undefined aliases
    sql_upper = sql.upper()
    
    # Extract defined table aliases
    defined_aliases = set()
    from_matches = re.findall(r'FROM\s+([A-Z_][A-Z0-9_]*)\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*)', sql_upper)
    for table_name, alias in from_matches:
        defined_aliases.add(alias)
    
    join_matches = re.findall(r'JOIN\s+([A-Z_][A-Z0-9_]*)\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*)', sql_upper)
    for table_name, alias in join_matches:
        defined_aliases.add(alias)
    
    # Find undefined table aliases in column references
    column_pattern = r'([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))'
    column_matches = re.findall(column_pattern, sql_upper)
    
    for full_column_ref, column_name in column_matches:
        table_alias = full_column_ref.split('.')[0]
        
        # If alias is undefined, try to fix it
        if table_alias not in defined_aliases:
            # Try to find a table that has this column and get its defined alias
            for table_name, columns in schema_columns.items():
                if column_name in columns:
                    # Find the defined alias for this table
                    defined_alias = None
                    for defined_table, defined_alias_name in [(t, a) for t, a in from_matches + join_matches]:
                        if defined_table == table_name:
                            defined_alias = defined_alias_name
                            break
                    
                    if defined_alias:
                        # Replace the undefined alias with the correct defined alias
                        import re
                        pattern = rf'\b{table_alias}\.{column_name}\b'
                        replacement = f"{defined_alias}.{column_name}"
                        corrected_sql = re.sub(pattern, replacement, corrected_sql, flags=re.IGNORECASE)
                        logger.info(f"Fixed undefined alias '{table_alias}' to '{defined_alias}' for column '{column_name}'")
                        break
    
    # 1. Fix missing dots in table aliases (e.g., "a3 PAYMENT_STATUS_FLAG" -> "a3.PAYMENT_STATUS_FLAG")
    # Pattern: table_alias + space + column_name (but not "FROM", "WHERE", "AND", "OR", etc.)
    reserved_words = {'FROM', 'WHERE', 'AND', 'OR', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'ON', 'AS', 'ORDER', 'GROUP', 'HAVING', 'SELECT', 'BY'}
    
    # Find patterns like "a3 PAYMENT_STATUS_FLAG" and fix to "a3.PAYMENT_STATUS_FLAG"
    # But exclude SQL keywords and common patterns
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+([A-Z_][A-Z0-9_]*)\b'
    def fix_missing_dot(match):
        alias = match.group(1)
        potential_column = match.group(2)
        
        # Check if this looks like a column name (all caps with underscores) and not a reserved word
        # Also check that the alias is not a SQL keyword
        if (potential_column.isupper() and '_' in potential_column and 
            potential_column not in reserved_words and 
            alias.upper() not in reserved_words and
            len(alias) <= 3):  # Table aliases are typically 1-3 characters
            return f'{alias}.{potential_column}'
        return match.group(0)
    
    corrected_sql = re.sub(pattern, fix_missing_dot, corrected_sql)
    
    # 2. Fix extra dots in BETWEEN clauses (e.g., "AND.TO_DATE" -> "AND TO_DATE")
    # This handles cases where LLM adds extra dots before function calls
    corrected_sql = re.sub(r'\bAND\.(TO_DATE|TO_CHAR|TO_NUMBER|COUNT|SUM|AVG|MIN|MAX)\b', r'AND \1', corrected_sql, flags=re.IGNORECASE)
    corrected_sql = re.sub(r'\bOR\.(TO_DATE|TO_CHAR|TO_NUMBER|COUNT|SUM|AVG|MIN|MAX)\b', r'OR \1', corrected_sql, flags=re.IGNORECASE)
    
    # 3. Fix case-insensitive column references to match schema exactly
    for table_name, columns in schema_columns.items():
        for column in columns:
            # Pattern to match any case of column name after table alias
            pattern = rf'\b([a-zA-Z_][a-zA-Z0-9_]*)\.{re.escape(column)}\b'
            def fix_column_case(match):
                alias = match.group(1)
                return f'{alias}.{column}'
            corrected_sql = re.sub(pattern, fix_column_case, corrected_sql, flags=re.IGNORECASE)
    
    # 4. Fix common typos dynamically by checking against schema
    # This is more robust than hardcoded corrections
    common_typos = {
        'TERS_DATE': 'TERMS_DATE',
        'TERM_DATE': 'TERMS_DATE', 
    }
    
    for typo, correct in common_typos.items():
        # Only fix if the correct column exists in schema
        if any(correct in columns for columns in schema_columns.values()):
            corrected_sql = corrected_sql.replace(typo, correct)
    
    return corrected_sql

def extract_sql_from_response(response: str, schema_columns: Dict[str, List[str]] = None) -> str:
    """
    Extract only the SQL query from the LLM response, removing any explanatory text.
    Enhanced to handle empty responses and provide better error messages.
    """
    # Check if response is empty or just whitespace/newlines
    if not response or not response.strip():
        logger.warning("Empty response from LLM model")
        return "-- ERROR: Empty response from LLM model"
    
    # Check for timeout error messages
    if "ERROR: Request timed out" in response or "timed out" in response.lower():
        return "-- ERROR: Request timed out"
    
    # Check for connection error messages
    if "ERROR: Cannot connect to Ollama" in response or "connection" in response.lower():
        return "-- ERROR: Cannot connect to Ollama"
    
    # CRITICAL: Check if LLM returned error messages instead of SQL
    if (response.startswith('-- ERROR:') or 
        'ERROR:' in response or 
        'SUGGESTIONS:' in response or
        'Column' in response and 'not found' in response or
        'not found in available schema' in response):
        logger.warning(f"LLM returned error message instead of SQL: {response[:200]}...")
        return "-- ERROR: LLM returned error message instead of SQL - retry needed"
    
    # ENHANCED: More robust SQL extraction
    logger.debug(f"Extracting SQL from response: {response[:200]}...")
    
    # Method 1: Look for SQL: section first (most common)
    if 'SQL:' in response:
        sql_section = response.split('SQL:')[1].strip()
        logger.debug(f"Found SQL: section: {sql_section[:100]}...")
        
        # Check if it contains markdown code blocks
        if '```sql' in sql_section:
            # Extract from markdown within SQL: section
            start_marker = '```sql'
            end_marker = '```'
            start_idx = sql_section.find(start_marker)
            if start_idx != -1:
                start_idx += len(start_marker)
                end_idx = sql_section.find(end_marker, start_idx)
                if end_idx != -1:
                    sql_content = sql_section[start_idx:end_idx].strip()
                    # Clean up any remaining markdown artifacts
                    sql_content = sql_content.replace('```', '').strip()
                    # Remove any trailing markdown markers or backticks
                    while sql_content.endswith('`'):
                        sql_content = sql_content.rstrip('`').strip()
                    if sql_content and sql_content.upper().startswith('SELECT'):
                        logger.info(f"Successfully extracted SQL from markdown: {sql_content[:100]}...")
                        # Apply dynamic corrections if schema columns are provided
                        if schema_columns:
                            sql_content = correct_sql_dynamically(sql_content, schema_columns)
                        return sql_content
        
        # Method 2: Direct SQL extraction from SQL: section without markdown
        lines = sql_section.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                in_sql = True
                sql_lines.append(line)
            elif in_sql and line:  # Continue SQL if we're already in SQL
                # Stop if we hit verification text or other non-SQL content
                if any(word in line.upper() for word in ['VERIFICATION:', 'EVERY', 'COLUMN', 'EXISTS', 'AVAILABLE', 'COMPATIBLE', 'STEP', 'ANALYZE', 'PLAN', 'WRITE', 'SELF-VERIFICATION', 'CHECKLIST', 'REASONING']):
                    break
                sql_lines.append(line)
            elif in_sql and not line:  # Keep empty lines within SQL
                sql_lines.append(line)
            elif in_sql and (line.upper().startswith('--') or line.upper().startswith('/*')):
                break  # Stop at comments
        
        result = '\n'.join(sql_lines).strip()
        if result and result.upper().startswith('SELECT'):
            logger.info(f"Successfully extracted SQL from SQL: section: {result[:100]}...")
            # Apply dynamic corrections if schema columns are provided
            if schema_columns:
                result = correct_sql_dynamically(result, schema_columns)
            return result
    
    # Method 3: Check for markdown code blocks with SQL (fallback)
    if '```sql' in response:
        logger.debug("Found ```sql markdown block")
        # Extract content between ```sql and ```
        start_marker = '```sql'
        end_marker = '```'
        start_idx = response.find(start_marker)
        if start_idx != -1:
            start_idx += len(start_marker)
            end_idx = response.find(end_marker, start_idx)
            if end_idx != -1:
                sql_content = response[start_idx:end_idx].strip()
                # Clean up any remaining markdown artifacts
                sql_content = sql_content.replace('```', '').strip()
                # Remove any trailing markdown markers or backticks
                while sql_content.endswith('`'):
                    sql_content = sql_content.rstrip('`').strip()
                # Remove any trailing ``` markers
                if sql_content.endswith('```'):
                    sql_content = sql_content[:-3].strip()
                if sql_content and sql_content.upper().startswith('SELECT'):
                    logger.info(f"Successfully extracted SQL from markdown block: {sql_content[:100]}...")
                    # Apply dynamic corrections if schema columns are provided
                    if schema_columns:
                        sql_content = correct_sql_dynamically(sql_content, schema_columns)
                    return sql_content
    
    # Method 4: Fallback - Find the first SELECT statement anywhere in response
    logger.debug("Using fallback method to find SELECT statement")
    lines = response.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        line = line.strip()
        if line.upper().startswith('SELECT'):
            in_sql = True
            sql_lines.append(line)
        elif in_sql:
            # Stop at explanatory text - only stop at clear non-SQL content
            if (line.upper().startswith('--') or 
                line.upper().startswith('/*') or
                line.upper().startswith('REASONING:') or
                line.upper().startswith('VERIFICATION:') or
                line.upper().startswith('NOTE:') or
                line.upper().startswith('EXPLANATION:') or
                line.upper().startswith('COMMENT:') or
                line.upper().startswith('ANALYSIS:') or
                line.upper().startswith('SUMMARY:')):
                break
            elif line:  # Non-empty line
                sql_lines.append(line)
            elif not line:  # Empty line
                sql_lines.append(line)
    
    result = '\n'.join(sql_lines).strip()
    
    # Enhanced validation: Check for corrupted/malformed SQL
    if not result or not result.upper().startswith('SELECT'):
        logger.warning(f"No valid SQL found in LLM response. Response preview: {response[:200]}...")
        return "-- ERROR: No valid SQL found in LLM response"
    
    # Check for corrupted SQL (contains random characters, incomplete statements)
    if ('=' * 10 in result or  # Multiple equal signs
        'Latest =' in result or  # Incomplete column aliases
        result.count('=') > 20 or  # Too many equal signs (likely corrupted)
        len(result.split('\n')) < 3):  # Too short to be complete SQL
        logger.warning(f"Detected corrupted SQL response: {result[:100]}...")
        return "-- ERROR: LLM generated corrupted SQL - retry needed"
    
    logger.info(f"Successfully extracted SQL using fallback method: {result[:100]}...")
    # Apply dynamic corrections if schema columns are provided
    if schema_columns:
        result = correct_sql_dynamically(result, schema_columns)
    
    return result

def parse_sql_table_column_mappings(sql: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse SQL using a robust, context-aware approach that handles complex SQL patterns.
    Uses multiple parsing strategies for maximum reliability.
    Returns dict mapping table_alias -> [(column_name, full_table_name), ...]
    """
    import re
    
    mappings = {}
    sql_upper = sql.upper()
    
    try:
        # Strategy 1: Extract table-alias mappings using comprehensive regex patterns
        table_alias_map = {}
        
        # Enhanced patterns that handle more SQL constructs
        patterns = [
            # Standard FROM/JOIN patterns
            r'FROM\s+([A-Z_][A-Z0-9_]*)\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*)',
            r'JOIN\s+([A-Z_][A-Z0-9_]*)\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*)',
            # Handle subqueries and CTEs
            r'WITH\s+([A-Z_][A-Z0-9_]*)\s+AS\s*\([^)]*\)',
            r'\)\s+([A-Z_][A-Z0-9_]*)',  # Subquery aliases like ") sub"
            # Handle UNION and other constructs
            r'UNION\s+SELECT[^F]*FROM\s+([A-Z_][A-Z0-9_]*)\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                if len(match) == 2:
                    table_name, alias = match
                    table_alias_map[alias] = table_name
                elif len(match) == 1:
                    # For patterns that only capture one group
                    alias = match
                    table_alias_map[alias] = alias  # Use alias as table name for CTEs
        
        # Strategy 2: Extract column references using multiple patterns
        column_patterns = [
            # Standard table_alias.column_name pattern
            r'([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))',
            # Handle column references in functions and expressions
            r'(?:SUM|COUNT|AVG|MIN|MAX|ROUND|TRUNC|TO_CHAR|TO_DATE|EXTRACT)\s*\(\s*([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))',
            # Handle CASE statements
            r'CASE\s+WHEN\s+([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))',
            # Handle ORDER BY, GROUP BY, HAVING clauses
            r'(?:ORDER\s+BY|GROUP\s+BY|HAVING)\s+([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))',
        ]
        
        for pattern in column_patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                if len(match) == 2:
                    full_column_ref, column_name = match
                    table_alias = full_column_ref.split('.')[0]
                    
                    # Get the actual table name from the alias
                    actual_table = table_alias_map.get(table_alias, table_alias)
                    
                    if table_alias not in mappings:
                        mappings[table_alias] = []
                    mappings[table_alias].append((column_name, actual_table))
        
        # Strategy 3: Fallback - if no mappings found, try simpler approach
        if not mappings:
            # Extract any table_alias.column_name patterns we can find
            simple_pattern = r'([A-Z_][A-Z0-9_]*\.([A-Z_][A-Z0-9_]*))'
            matches = re.findall(simple_pattern, sql_upper)
            
            for full_column_ref, column_name in matches:
                table_alias = full_column_ref.split('.')[0]
                
                # Try to infer table name from alias or use alias as table name
                actual_table = table_alias_map.get(table_alias, table_alias)
                
                if table_alias not in mappings:
                    mappings[table_alias] = []
                mappings[table_alias].append((column_name, actual_table))
        
        return mappings
        
    except Exception as e:
        logger.warning(f"Failed to parse SQL for table-column mappings: {e}")
        return {}

def validate_table_column_relationships(sql: str, allowed_columns_by_table: Dict[str, List[str]]) -> List[str]:
    """
    Validate that each column is used with the correct table using simple regex patterns.
    Returns list of violation messages.
    """
    violations = []
    
    try:
        # Parse SQL to get table-column mappings using our robust regex approach
        mappings = parse_sql_table_column_mappings(sql)
        
        for table_alias, column_mappings in mappings.items():
            for column_name, full_table_name in column_mappings:
                # Check if column exists in the specified table
                if full_table_name in allowed_columns_by_table:
                    allowed_columns = allowed_columns_by_table[full_table_name]
                    if column_name not in allowed_columns:
                        violations.append(
                            f"Column '{column_name}' does not exist in table '{full_table_name}' "
                            f"(used with alias '{table_alias}')"
                        )
                else:
                    violations.append(
                        f"Table '{full_table_name}' not found in schema "
                        f"(used with alias '{table_alias}')"
                    )
        
        return violations
        
    except Exception as e:
        logger.warning(f"Error in table-column relationship validation: {e}")
        return []  # Return empty violations list on error - don't fail validation

def validate_sql_against_schema(sql: str, available_columns: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Validate generated SQL against available schema to catch hallucinated columns.
    Returns validation result with suggestions.
    IMPORTANT: This function only validates - it does NOT modify the original SQL.
    """
    import re
    
    validation_result = {
        "ok": True,
        "errors": [],
        "suggestions": []
    }
    
    # Extract column names from SQL, handling table aliases properly
    # More sophisticated pattern to avoid false positives
    # Look for patterns like: table_alias.COLUMN_NAME, table.COLUMN_NAME, or standalone COLUMN_NAME
    # But exclude table names, aliases, and string literals
    
    # CRITICAL FIX: Create a separate cleaned version ONLY for column extraction
    # This does NOT modify the original SQL - it's only used for validation
    sql_for_validation = sql.upper()  # Work with uppercase copy
    sql_for_validation = re.sub(r"'[^']*'", '', sql_for_validation)  # Remove string literals for column extraction
    sql_for_validation = re.sub(r'--.*$', '', sql_for_validation, flags=re.MULTILINE)  # Remove comments
    
    # More sophisticated pattern to extract only actual column references
    # Look for patterns like: table_alias.COLUMN_NAME or just COLUMN_NAME
    # But exclude table names, parameters, and SQL keywords
    
    # First, remove parameter placeholders (like :amount_threshold)
    sql_for_validation = re.sub(r':[A-Z_][A-Z0-9_]*', '', sql_for_validation)
    
    # Don't modify function calls - they're needed for proper SQL structure
    # The issue was that we were corrupting the SQL syntax
    # sql_clean = re.sub(r'\b(COUNT|SUM|AVG|MIN|MAX|ROUND|TRUNC|SUBSTR|UPPER|LOWER|LENGTH|REPLACE|DECODE|COALESCE|NVL|TO_CHAR|TO_NUMBER|TO_DATE|CAST|CONVERT)\s*\(', 'FUNC(', sql_clean, flags=re.IGNORECASE)
    
    # Remove AS aliases more carefully - only remove the AS keyword and alias name
    # This prevents corruption of the SQL structure
    # Don't remove AS aliases - they're needed for proper SQL structure
    # sql_clean = re.sub(r'\s+AS\s+([A-Z_][A-Z0-9_]*)', r' \1', sql_clean, flags=re.IGNORECASE)
    
    # Extract table aliases from FROM and JOIN clauses to exclude them from column validation
    table_aliases = set()
    
    # Find table aliases in FROM clause: TABLE_NAME alias
    from_matches = re.findall(r'FROM\s+([A-Z_][A-Z0-9_]*)\s+([A-Z_][A-Z0-9_]*)', sql_for_validation)
    for table, alias in from_matches:
        table_aliases.add(alias)
    
    # Find table aliases in JOIN clauses: JOIN TABLE_NAME alias
    join_matches = re.findall(r'JOIN\s+([A-Z_][A-Z0-9_]*)\s+([A-Z_][A-Z0-9_]*)', sql_for_validation)
    for table, alias in join_matches:
        table_aliases.add(alias)
    
    # ENHANCED: Extract only actual column references, not aliases or table names
    potential_columns = set()
    
    # Extract columns from SELECT clause (the most important part)
    select_pattern = r'SELECT\s+(.*?)\s+FROM'
    select_match = re.search(select_pattern, sql_for_validation, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        
        # Split by comma and process each column expression
        column_expressions = [expr.strip() for expr in select_clause.split(',')]
        
        for expr in column_expressions:
            # Handle AS aliases - extract the actual column, not the alias
            if ' AS ' in expr.upper():
                parts = expr.upper().split(' AS ')
                if len(parts) == 2:
                    left_part = parts[0].strip()
                    # Extract actual columns from the left part (before AS)
                    actual_columns = re.findall(r'([A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*)', left_part)
                    for col in actual_columns:
                        potential_columns.add(col)
                    continue
            
            # Handle simple columns without aliases
            actual_columns = re.findall(r'([A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*)', expr.upper())
            for col in actual_columns:
                potential_columns.add(col)
    
    # Extract columns from WHERE clause
    where_pattern = r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)'
    where_match = re.search(where_pattern, sql_for_validation, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        where_columns = re.findall(r'([A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*)', where_clause, flags=re.IGNORECASE)
        for col in where_columns:
            potential_columns.add(col.upper())
    
    # Extract columns from JOIN conditions
    join_pattern = r'JOIN\s+[A-Z_][A-Z0-9_]*\s+[A-Z_][A-Z0-9_]*\s+ON\s+([^)]+)'
    join_matches = re.findall(join_pattern, sql_for_validation, flags=re.IGNORECASE)
    for join_condition in join_matches:
        join_columns = re.findall(r'([A-Z_][A-Z0-9_]*\.[A-Z_][A-Z0-9_]*)', join_condition, flags=re.IGNORECASE)
        for col in join_columns:
            potential_columns.add(col.upper())
    
    # Convert set to list for processing
    potential_columns = list(potential_columns)
    
    # Remove aliases that appear in ORDER BY, GROUP BY, HAVING clauses
    # These are not actual column references but aliases
    order_by_aliases = set()
    group_by_aliases = set()
    
    # Extract aliases from ORDER BY clause
    order_by_matches = re.findall(r'ORDER\s+BY\s+([A-Z_][A-Z0-9_]*)', sql_for_validation, flags=re.IGNORECASE)
    for alias in order_by_matches:
        order_by_aliases.add(alias.upper())
    
    # Extract aliases from GROUP BY clause  
    group_by_matches = re.findall(r'GROUP\s+BY\s+([A-Z_][A-Z0-9_]*)', sql_for_validation, flags=re.IGNORECASE)
    for alias in group_by_matches:
        group_by_aliases.add(alias.upper())
    
    # Filter out common SQL keywords, table names, and single-letter aliases
    sql_keywords = {
        # SQL Keywords
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'BETWEEN', 
        'INNER', 'LEFT', 'RIGHT', 'OUTER', 'GROUP', 'BY', 'ORDER', 'HAVING', 
        'UNION', 'ALL', 'DISTINCT', 'AS', 'IS', 'NULL', 'NOT', 'IN', 'EXISTS', 
        'LIKE', 'ESCAPE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IF',
        
        # SQL Functions
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'TRUNC', 'SUBSTR',
        'UPPER', 'LOWER', 'LENGTH', 'REPLACE', 'DECODE', 'COALESCE', 'NVL', 
        'TO_CHAR', 'TO_NUMBER', 'TO_DATE', 'CAST', 'CONVERT',
        
        # Date/Time Functions
        'YYYY', 'MM', 'DD', 'HH24', 'MI', 'SS', 'MONTH', 'YEAR', 'DAY',
        'SYSDATE', 'CURRENT_DATE', 'CURRENT_TIMESTAMP',
        
        # Oracle-specific functions
        'ROWID', 'ROWNUM', 'LEVEL', 'CONNECT_BY_ROOT', 'PRIOR',
        
        # Oracle 12c+ syntax keywords
        'FETCH', 'FIRST', 'ROWS', 'ONLY', 'ROW', 'DESC', 'ASC'
    }
    
    # Common table name patterns to exclude (generic patterns)
    table_patterns = [
        r'^[A-Z_]+_ALL$',  # Tables ending with _ALL
        r'^[A-Z_]+_B$',    # Tables ending with _B
        r'^[A-Z_]+_V$',    # Tables ending with _V
        r'^[A-Z_]+_S$',    # Tables ending with _S
        r'^[A-Z_]+_VW$',   # Tables ending with _VW
        r'^[A-Z_]+_MV$',   # Tables ending with _MV
        r'^[A-Z_]+_SUMMARY$',  # Summary tables
        r'^[A-Z_]+_HISTORY$',  # History tables
        r'^[A-Z_]+_ARCHIVE$',  # Archive tables
        r'^[A-Z_]+_TEMP$',     # Temporary tables
        r'^[A-Z_]+_BKP$',      # Backup tables
    ]
    
    # ENHANCED: Process extracted columns properly
    sql_columns = []
    for col in potential_columns:
        # Skip if empty
        if not col or not col.strip():
            continue
            
        # Skip if it's just a table alias (single letter)
        if len(col) == 1 and col.isalpha():
            continue
            
        # Skip if it's a table name (no dot in it)
        if '.' not in col:
            continue
            
        # Extract the column name part (after the dot)
        if '.' in col:
            column_name = col.split('.')[1]
            
            # Skip SQL keywords
            if column_name in sql_keywords:
                continue
                
            # Skip aliases that appear in ORDER BY, GROUP BY
            if column_name in order_by_aliases:
                continue
            if column_name in group_by_aliases:
                continue
            
            # Skip table name patterns
            is_table = False
            for pattern in table_patterns:
                if re.match(pattern, column_name):
                    is_table = True
                    break
            if is_table:
                continue
            
            # This is a valid column reference to validate
            sql_columns.append(col)
    
    # Debug: Log the cleaned SQL and extracted columns
    logger.debug(f"SQL for validation (cleaned for column extraction): {sql_for_validation[:500]}...")
    logger.debug(f"Extracted columns for validation: {sql_columns}")
    
    # Check each column against available schema
    for col in sql_columns:
        found = False
        suggestions = []
        
        # Extract table and column from the full column reference (e.g., "A.PERIOD_NAME")
        if '.' in col:
            table_alias, column_name = col.split('.', 1)
            
            # Find the actual table name from the alias
            actual_table = None
            
            # First, try to find the table by analyzing the SQL structure
            # Look for patterns like: FROM GL_JE_HEADERS A or JOIN GL_JE_LINES B
            from_pattern = rf'FROM\s+([A-Z_][A-Z0-9_]*)\s+{table_alias.upper()}'
            join_pattern = rf'JOIN\s+([A-Z_][A-Z0-9_]*)\s+{table_alias.upper()}'
            
            from_match = re.search(from_pattern, sql_for_validation)
            join_match = re.search(join_pattern, sql_for_validation)
            
            if from_match:
                actual_table = from_match.group(1)
            elif join_match:
                actual_table = join_match.group(1)
            else:
                # CRITICAL: Check if table alias is undefined
                # This is the root cause of the issue - LLM generating SQL with undefined aliases
                validation_result["ok"] = False
                validation_result["errors"].append(f"Undefined table alias '{table_alias}' used in column reference '{col}'. Available aliases: {list(table_aliases)}")
                continue
                
                # Fallback: try to match by common alias patterns (disabled to catch undefined aliases)
                # Map common aliases to likely tables based on context
                if table_alias.upper() == 'A':
                    # Usually the first table (FROM clause)
                    for table, columns in available_columns.items():
                        if 'HEADER' in table.upper() or 'BATCH' in table.upper():
                            actual_table = table
                            break
                elif table_alias.upper() == 'B':
                    # Usually the second table (JOIN clause)
                    for table, columns in available_columns.items():
                        if 'LINE' in table.upper() or 'DETAIL' in table.upper():
                            actual_table = table
                            break
                elif table_alias.upper() == 'C':
                    # Usually the third table
                    for table, columns in available_columns.items():
                        if 'CODE' in table.upper() or 'COMBINATION' in table.upper():
                            actual_table = table
                            break
                
                # If still not found, try exact match
                if not actual_table:
                    for table, columns in available_columns.items():
                        if table_alias.upper() == table.upper():
                            actual_table = table
                            break
            
            if actual_table:
                # Check if the column exists in this table
                table_columns = available_columns.get(actual_table, [])
                if column_name.upper() in [c.upper() for c in table_columns]:
                    found = True
                else:
                    # Find similar columns in this table
                    for available_col in table_columns:
                        if (column_name.upper() in available_col.upper() or 
                            available_col.upper() in column_name.upper() or
                            abs(len(column_name) - len(available_col)) <= 2):
                            suggestions.append(f"{actual_table}.{available_col}")
            else:
                # Table not found, suggest all tables
                for table, columns in available_columns.items():
                    for available_col in columns:
                        if column_name.upper() in available_col.upper():
                            suggestions.append(f"{table}.{available_col}")
        
        # CRITICAL FIX: Only skip validation for actual SQL keywords and functions, NOT for column names
        # The previous logic was incorrectly skipping validation for legitimate column names
        sql_keywords_and_functions = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'BETWEEN', 'IN', 'EXISTS',
            'GROUP', 'BY', 'ORDER', 'HAVING', 'UNION', 'ALL', 'DISTINCT', 'AS', 'IS', 'NULL',
            'NOT', 'LIKE', 'ESCAPE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'IF',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'TRUNC', 'SUBSTR', 'UPPER', 'LOWER',
            'LENGTH', 'REPLACE', 'DECODE', 'COALESCE', 'NVL', 'TO_CHAR', 'TO_NUMBER', 'TO_DATE',
            'CAST', 'CONVERT', 'EXTRACT', 'SYSDATE', 'CURRENT_DATE', 'CURRENT_TIMESTAMP',
            'ROWID', 'ROWNUM', 'LEVEL', 'CONNECT_BY_ROOT', 'PRIOR', 'FETCH', 'FIRST', 'ROWS',
            'ONLY', 'ROW', 'DESC', 'ASC', 'YYYY', 'MM', 'DD', 'HH24', 'MI', 'SS', 'MONTH', 'YEAR', 'DAY', 'HH', 'AM', 'PM'
        }
        
        # Only skip validation for actual SQL keywords/functions, not for column names
        if not found and column_name not in sql_keywords_and_functions:
            validation_result["ok"] = False
            validation_result["errors"].append(f"Column '{col}' not found in available schema")
            if suggestions:
                validation_result["suggestions"].append(f"Column '{col}' not found. Similar columns: {', '.join(suggestions[:3])}")
    
    # Additional check for invalid JOIN logic
    # Look for patterns like table1.column1 = table2.column2 where the columns are incompatible
    join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
    join_matches = re.findall(join_pattern, sql_for_validation)
    
    for table1, col1, table2, col2 in join_matches:
        # Check if these are valid columns in their respective tables
        table1_cols = available_columns.get(table1.upper(), [])
        table2_cols = available_columns.get(table2.upper(), [])
        
        if table1_cols and table2_cols:
            col1_found = col1.upper() in [c.upper() for c in table1_cols]
            col2_found = col2.upper() in [c.upper() for c in table2_cols]
            
            if not col1_found:
                validation_result["ok"] = False
                validation_result["errors"].append(f"Column '{col1}' not found in table '{table1}'")
            if not col2_found:
                validation_result["ok"] = False
                validation_result["errors"].append(f"Column '{col2}' not found in table '{table2}'")
            
            # Check for obviously incompatible JOIN logic
            if col1_found and col2_found:
                # Check for common JOIN logic errors
                if (col1.upper() == 'CUSTOMER_TRX_ID' and col2.upper() == 'DATA_SET_ID') or \
                   (col1.upper() == 'DATA_SET_ID' and col2.upper() == 'CUSTOMER_TRX_ID'):
                    validation_result["ok"] = False
                    validation_result["errors"].append(f"Invalid JOIN: {col1} and {col2} are incompatible identifiers")
                    validation_result["suggestions"].append(f"Use compatible ID columns for JOINs (e.g., CUSTOMER_TRX_ID with CUSTOMER_TRX_ID)")
    
    return validation_result

def call_openai_api(system_prompt: str, user_prompt: str, model: str = OPENAI_MODEL) -> str:
    """
    Calls OpenAI API with better error handling and debugging.
    """
    if not openai_client:
        return "-- ERROR: OpenAI client not available. Check API key and package installation."
    
    logger.info(f"Calling OpenAI with model: {model}")
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        text = response.choices[0].message.content
        logger.info(f"Received response from OpenAI: {text[:100]}...")
        return text.strip()
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return f"-- ERROR: OpenAI API call failed: {e}"

def call_ollama_rest(system_prompt: str, user_prompt: str, model: str = "sqlcoder:15b", timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Calls Ollama local REST API /api/chat with better error handling and debugging.
    """
    logger.info(f"Calling Ollama with model: {model}")
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40
        }
    }
    
    try:
        logger.debug(f"Sending request to Ollama with payload: {json.dumps(payload, indent=2)}")
        resp = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        
        if resp.status_code != 200:
            logger.error(f"Ollama returned status {resp.status_code}: {resp.text[:400]}")
            return f"-- ERROR: Ollama returned status {resp.status_code}: {resp.text[:400]}"
        
        # Parse response with better error handling
        try:
            j = resp.json()
            logger.debug(f"Raw response from Ollama: {json.dumps(j, indent=2)}")
            
            # Handle different response formats
            if "message" in j and "content" in j["message"]:
                text = j["message"]["content"]
            elif "response" in j:
                text = j["response"]
            elif "choices" in j and len(j["choices"]) > 0 and "message" in j["choices"][0]:
                text = j["choices"][0]["message"]["content"]
            else:
                text = resp.text
                
            logger.info(f"Received response from Ollama: {text[:100]}...")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            logger.error(f"Raw response: {resp.text}")
            return f"-- ERROR: Failed to parse response: {e}"
            
    except requests.exceptions.Timeout:
        logger.error(f"Ollama request timed out after {timeout} seconds")
        return f"-- ERROR: Request timed out after {timeout} seconds. Try using a smaller model or reducing prompt complexity."
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        return "-- ERROR: Cannot connect to Ollama. Please ensure it's running on localhost:11434"
    except Exception as e:
        logger.exception("Ollama REST call failed")
        return f"-- ERROR: Ollama request failed: {e}"

# ---------- SQL parse / verifier ----------
def extract_tables_and_columns_from_sql(sql_text: str) -> Dict[str, Any]:
    try:
        parsed = sqlglot.parse_one(sql_text, read="oracle")
    except Exception as e:
        return {"error": f"sqlglot parse error: {e}"}
    
    tables = set()
    columns = set()
    
    # Extract tables
    for t in parsed.find_all(exp.Table):
        name = getattr(t, "this", None) or getattr(t, "this_", None) or getattr(t, "name", None)
        if hasattr(name, "name"):
            tn = name.name
        else:
            tn = str(name) if name is not None else None
        if tn:
            tables.add(tn.upper())
    
    # Extract columns
    for c in parsed.find_all(exp.Column):
        col_name = getattr(c, "this", None) or getattr(c, "name", None) or getattr(c, "this_", None)
        if hasattr(col_name, "name"):
            cn = col_name.name
        else:
            cn = str(col_name) if col_name is not None else None
        if cn:
            columns.add(cn.upper())
    
    return {"tables": sorted(tables), "columns": sorted(columns)}

def post_sql_check(sql_text: str, allowed_tables_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures every table and column used in SQL exists in allowed_tables_summary.
    Enhanced to prevent column hallucination by checking column-table relationships.
    """
    extracted = extract_tables_and_columns_from_sql(sql_text)
    if "error" in extracted:
        return {"ok": False, "error": extracted["error"]}
    
    used_tables = set(extracted["tables"])
    used_columns = set(extracted["columns"])

    allowed_tables = set([t.upper() for t in allowed_tables_summary.keys()])
    allowed_columns_by_table = {t.upper(): set([c.upper() for c in allowed_tables_summary[t].get("columns",[])]) for t in allowed_tables_summary}

    # Check for unknown tables
    unknown_tables = sorted(list(used_tables - allowed_tables))
    
    # Check for columns that don't exist in any table
    allowed_columns_all = set()
    for s in allowed_columns_by_table.values():
        allowed_columns_all |= s
    
    # Special handling for * (wildcard) - this is not allowed
    if '*' in used_columns:
        unknown_columns = ['*'] + sorted([col for col in used_columns if col != '*' and col not in allowed_columns_all])
    else:
        unknown_columns = sorted([col for col in used_columns if col not in allowed_columns_all])

    # ENHANCED: Check for column-table relationship violations
    # Now properly validates that each column belongs to its referenced table
    column_table_violations = validate_table_column_relationships(sql_text, allowed_columns_by_table)

    # Overall validation
    ok = (len(unknown_tables) == 0 and len(unknown_columns) == 0 and len(column_table_violations) == 0)
    
    logger.info(f"SQL validation - Tables: {used_tables}, Columns: {used_columns}")
    logger.info(f"Unknown tables: {unknown_tables}, Unknown columns: {unknown_columns}")
    if column_table_violations:
        logger.warning(f"Column-table violations: {column_table_violations}")
    
    return {
        "ok": ok,
        "used_tables": sorted(list(used_tables)),
        "used_columns": sorted(list(used_columns)),
        "unknown_tables": unknown_tables,
        "unknown_columns": unknown_columns,
        "column_table_violations": column_table_violations
    }

# ---------- validation + top-level ----------
def validate_sql(sql_text: str) -> Dict[str,Any]:
    try:
        parsed = sqlglot.parse_one(sql_text, read="oracle")
        normalized = parsed.sql(dialect="oracle")
        return {"ok": True, "normalized": normalized}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def generate_sql_from_text_semantic(user_query: str, chroma_k: int = 50, model: str = None) -> Dict[str,Any]:
    """
    Enhanced SQL generation that focuses on semantic meaning from comments.
    """
    logger.info(f"Starting semantic SQL generation for query: {user_query}")
    
    # Auto-select best model if none specified
    if model is None:
        if OPENAI_API_KEY:
            model = "gpt-4o-mini"  # Fast and reliable
            logger.info("Auto-selected OpenAI gpt-4o-mini for optimal performance")
        else:
            model = "llama3:8b"  # Fastest Ollama model
            logger.warning("Auto-selected Ollama llama3:8b (OpenAI not available - may be slower)")
    
    # 1) Retrieve documents with semantic focus - increase k for better column coverage
    effective_k = max(chroma_k, 100)  # Ensure minimum 100 documents for column coverage in Pinecone mode
    r = retrieve_docs_semantic(user_query, k=effective_k)
    docs = r["docs"]
    if not docs:
        logger.error("No semantically relevant documents retrieved from Chroma")
        return {"error": "No semantically relevant schema information found for this query."}

    # 2) Summarize relevant tables/columns with semantic focus
    table_summary = summarize_relevant_tables(docs, user_query)

    # 3) Build semantic-focused prompt
    # Add available tables list to prevent hallucination
    available_tables = list(table_summary.keys())
    prompt = build_semantic_prompt(user_query, table_summary, available_tables)

    # 4) Call LLM (OpenAI or Ollama) with fallback mechanism
    if model.startswith("gpt-") or model.startswith("o1-"):
        # Use OpenAI for GPT models
        llm_response = call_openai_api(SYSTEM_PROMPT, prompt, model=model)
    else:
        # Use Ollama for local models
        llm_response = call_ollama_rest(SYSTEM_PROMPT, prompt, model=model)
    
    # Debug: Log the full LLM response
    logger.debug(f"Full LLM response: {llm_response[:1000]}...")
    
    # Create schema columns dict for SQL correction
    schema_columns = {}
    for table_name, table_data in table_summary.items():
        schema_columns[table_name] = [col[0] for col in table_data.get('columns', [])]
    
    llm_sql = extract_sql_from_response(llm_response, schema_columns)
    
    # 4.5) Validate SQL against schema to prevent hallucination
    if llm_sql and not llm_sql.startswith("-- ERROR:"):
        available_columns = {}
        for table_name, table_data in table_summary.items():
            available_columns[table_name] = table_data.get("columns", [])
        
        schema_validation = validate_sql_against_schema(llm_sql, available_columns)
        if not schema_validation["ok"]:
            logger.warning(f"LLM generated SQL with invalid columns: {schema_validation['errors']}")
            error_msg = "; ".join(schema_validation["errors"])
            if schema_validation["suggestions"]:
                error_msg += "\n\nSUGGESTIONS:\n" + "\n".join(schema_validation["suggestions"])
            llm_sql = f"-- ERROR: {error_msg}"
    
    # 5) If timeout, connection error, or LLM returned error message instead of SQL, try with a smaller model
    if llm_sql and llm_sql.startswith("-- ERROR:") and ("timed out" in llm_sql or "failed" in llm_sql or "LLM returned error message instead of SQL" in llm_sql):
        logger.warning(f"Model {model} failed, trying with fallback models")
        
        # Choose fallback models based on original model type
        if model.startswith("gpt-") or model.startswith("o1-"):
            # OpenAI fallbacks
            fallback_models = ["gpt-4o-mini", "gpt-3.5-turbo"]
        else:
            # Ollama fallbacks
            fallback_models = ["llama3:8b", "sqlcoder:7b", "mistral:instruct"]
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                
                # If the original issue was LLM returning error messages, enhance the prompt
                enhanced_prompt = prompt
                if "LLM returned error message instead of SQL" in llm_sql:
                    enhanced_prompt = prompt + "\n\nCRITICAL: You must generate valid SQL using only the columns listed in ALL_AVAILABLE_COLUMNS above. Do NOT return error messages - generate working SQL even if you need to use different column names than you initially thought."
                
                if fallback_model.startswith("gpt-") or fallback_model.startswith("o1-"):
                    llm_response = call_openai_api(SYSTEM_PROMPT, enhanced_prompt, model=fallback_model)
                else:
                    llm_response = call_ollama_rest(SYSTEM_PROMPT, enhanced_prompt, model=fallback_model, timeout=120)
                llm_sql = extract_sql_from_response(llm_response, schema_columns)
                if not llm_sql.startswith("-- ERROR:"):
                    logger.info(f"Successfully generated SQL with fallback model: {fallback_model}")
                    break
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} also failed: {e}")
                continue
    
    # Note: Schema validation is already handled above by validate_sql_against_schema()
    # No need for duplicate validation that overrides error messages

    # 5) Validate SQL syntax
    validation = validate_sql(llm_sql)

    # 6) Post-SQL check against allowed schema
    post_check = {}
    if validation.get("ok"):
        post_check = post_sql_check(validation["normalized"], table_summary)
    else:
        # attempt to still run check on raw llm_sql if parse failed
        post_check = post_sql_check(llm_sql, table_summary)

    return {
        "relevant_tables": table_summary,
        "prompt_sent": prompt,
        "llm_sql": llm_sql,
        "validation": validation,
        "post_check": post_check
    }

# ============================================================================
# LANGCHAIN INTEGRATION
# ============================================================================

class OracleSQLChain:
    """
    LangChain-based SQL generation system integrated with existing ChromaDB and process docs.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain packages not available. Install with: pip install langchain langchain-community langchain-openai langchain-ollama")
        
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.chains = self._create_chains()
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on model name."""
        if self.model_name.startswith("gpt-") or self.model_name.startswith("o1-"):
            return ChatOpenAI(model=self.model_name, temperature=0)
        else:
            return Ollama(model=self.model_name, temperature=0)
    
    def _create_chains(self):
        """Create focused chains for different aspects of SQL generation."""
        return {
            "schema_analyzer": self._create_schema_analysis_chain(),
            "column_selector": self._create_column_selection_chain(),
            "sql_generator": self._create_sql_generation_chain(),
            "sql_validator": self._create_sql_validation_chain(),
            "sql_corrector": self._create_sql_correction_chain(),
            "date_function_fixer": self._create_date_function_fixer_chain()
        }
    
    def _create_schema_analysis_chain(self):
        """Chain 1: Analyze schema and identify relevant tables."""
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following Oracle database schema and user query to identify the most relevant tables.
        
        User Query: {user_query}
        
        Available Tables and Columns:
        {schema_info}
        
        Business Process Context:
        {process_context}
        
        Instructions:
        1. Identify which tables are most relevant to the query
        2. Explain why each table is relevant based on column descriptions and table purposes
        3. Rank tables by relevance (1-5 scale)
        4. Identify any missing tables that might be needed
        
        Output format:
        RELEVANT_TABLES:
        - table_name (relevance_score): reason_for_relevance
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_column_selection_chain(self):
        """Chain 2: Select appropriate columns based on query requirements."""
        prompt = ChatPromptTemplate.from_template("""
        Based on the user query and relevant tables, select the appropriate columns.
        
        User Query: {user_query}
        Relevant Tables: {relevant_tables}
        Available Columns: {available_columns}
        
        Instructions:
        1. Identify which columns are needed for the query
        2. For calculated values (like "total amount"), identify the base columns to use
        3. Ensure all selected columns actually exist in the schema
        4. Map query requirements to actual column names
        
        SEMANTIC UNDERSTANDING:
        5. When query asks for "per customer", look for CUSTOMER_ID columns, not transaction IDs
        6. When query asks for "per supplier", look for VENDOR_ID columns, not transaction IDs
        7. When query asks for "per organization", look for ORG_ID columns, not transaction IDs
        8. If query mentions customer names, include customer name columns
        9. If query mentions supplier names, include supplier name columns
        10. Analyze the business context to determine the correct identifier columns
        
        Output format:
        SELECTED_COLUMNS:
        - table_name.column_name: purpose_in_query
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_sql_generation_chain(self):
        """Chain 3: Generate the actual SQL query."""
        prompt = ChatPromptTemplate.from_template("""
        Generate a valid Oracle SQL query using the selected tables and columns.
        
        User Query: {user_query}
        Selected Tables: {selected_tables}
        Selected Columns: {selected_columns}
        Foreign Key Relationships: {foreign_keys}
        
        CRITICAL RULES:
        1. ALWAYS generate valid SQL - NEVER return error messages
        2. Use table aliases: a, b, c
        3. For dates: TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')
        4. Use only the columns that were selected in the previous step
        5. For aggregations, use appropriate SQL functions (SUM, COUNT, etc.)
        6. Use proper JOIN syntax based on foreign key relationships
        
        SEMANTIC UNDERSTANDING RULES:
        7. When query asks for "per customer", use actual customer identifier (CUSTOMER_ID) not transaction ID
        8. When query asks for "per supplier", use actual supplier identifier (VENDOR_ID) not transaction ID
        9. When query asks for "per organization", use actual organization identifier (ORG_ID) not transaction ID
        10. Analyze the query intent to determine the correct grouping column
        11. If you need customer names, join with customer tables using CUSTOMER_ID
        12. If you need supplier names, join with supplier tables using VENDOR_ID
        
        Output format:
        SQL:
        [Your valid Oracle SQL query]
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_sql_validation_chain(self):
        """Chain 4: Validate the generated SQL."""
        prompt = ChatPromptTemplate.from_template("""
        Validate the following SQL query for Oracle syntax and schema compliance.
        
        Generated SQL: {generated_sql}
        Schema: {schema_info}
        
        Instructions:
        1. Check for syntax errors
        2. Verify all table and column names exist in the provided schema
        3. Check for proper JOIN conditions
        4. Validate data type usage based on column types in schema
        5. CRITICAL: Distinguish between actual columns and SQL aliases (AS clauses)
        6. Check for malformed TO_DATE() functions with missing parameters
        7. Verify table selection matches query intent based on table purposes and column descriptions from schema
        
        Output format:
        VALIDATION_RESULT: [PASS/FAIL]
        ISSUES: [List any issues found, or "None" if valid]
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_sql_correction_chain(self):
        """Chain 5: Correct any issues found in the SQL."""
        prompt = ChatPromptTemplate.from_template("""
        Correct the following SQL query based on the validation results.
        
        Original SQL: {original_sql}
        Validation Issues: {validation_issues}
        Schema: {schema_info}
        
        Instructions:
        1. Fix all syntax errors
        2. Replace non-existent columns with correct ones from the provided schema
        3. Fix data type mismatches based on column types in schema
        4. Fix malformed TO_DATE() functions - ensure they have proper date strings and format masks
        5. CRITICAL: Do NOT treat SQL aliases (AS clauses) as invalid columns - they are valid
        6. Use the most appropriate table based on the query intent and table purposes from schema
        7. Ensure the query will run successfully using only columns and tables from the provided schema
        
        Output format:
        CORRECTED_SQL:
        [Your corrected Oracle SQL query]
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_date_function_fixer_chain(self):
        """Chain 6: Specifically fix malformed TO_DATE functions."""
        prompt = ChatPromptTemplate.from_template("""
        Fix malformed TO_DATE functions in the following SQL query.
        
        SQL Query: {sql_query}
        User Query Context: {user_query}
        
        Instructions:
        1. Find any TO_DATE(, ) functions with missing parameters
        2. Replace them with proper TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD') format
        3. Analyze the user query context to determine appropriate date ranges
        4. Extract date information from the user query (months, quarters, years, specific dates)
        5. Calculate the correct start and end dates based on the query requirements
        6. Use standard Oracle date format: TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')
        7. For month queries, use first day of month and first day of next month
        8. For quarter queries, use first day of quarter and first day of next quarter
        9. For year queries, use first day of year and first day of next year
        
        Output format:
        FIXED_SQL:
        [Your SQL query with corrected TO_DATE functions]
        """)
        
        return prompt | self.llm | StrOutputParser()
    
    def generate_sql_with_chromadb(self, user_query: str, chroma_k: int = RETRIEVE_K) -> Dict[str, Any]:
        """
        Generate SQL using LangChain chains with real ChromaDB schema and process context.
        """
        try:
            logger.info(f"Starting LangChain SQL generation for query: {user_query}")
            
            # Get schema information from ChromaDB (same as existing system)
            # First retrieve documents, then summarize tables, then build the prompt
            r = retrieve_docs_semantic(user_query, k=chroma_k)
            docs = r["docs"]
            if not docs:
                raise Exception("No semantically relevant documents retrieved from Chroma")
            
            table_summary = summarize_relevant_tables(docs, user_query)
            schema_info = build_semantic_prompt(user_query, table_summary)
            
            # Get process context (same as existing system)
            process_context = get_process_context_for_query(user_query)
            
            logger.info(f"Schema info length: {len(schema_info) if schema_info else 0}")
            logger.info(f"Process context length: {len(process_context) if process_context else 0}")
            
            logger.info(f"Retrieved schema info and process context for LangChain processing")
            
            # Step 1: Analyze schema and identify relevant tables
            relevant_tables = self.chains["schema_analyzer"].invoke({
                "user_query": user_query,
                "schema_info": schema_info,
                "process_context": process_context
            })
            
            # Step 2: Select appropriate columns
            selected_columns = self.chains["column_selector"].invoke({
                "user_query": user_query,
                "relevant_tables": relevant_tables,
                "available_columns": schema_info
            })
            
            # Step 3: Generate SQL
            generated_sql = self.chains["sql_generator"].invoke({
                "user_query": user_query,
                "selected_tables": relevant_tables,
                "selected_columns": selected_columns,
                "foreign_keys": schema_info  # This would be extracted from schema
            })
            
            # Step 4: Validate SQL
            validation_result = self.chains["sql_validator"].invoke({
                "generated_sql": generated_sql,
                "schema_info": schema_info
            })
            
            # Step 5: Fix TO_DATE functions if needed
            if "TO_DATE(, )" in generated_sql or "TO_DATE( , )" in generated_sql:
                generated_sql = self.chains["date_function_fixer"].invoke({
                    "sql_query": generated_sql,
                    "user_query": user_query
                })
            
            # Step 6: Correct if validation failed
            if "FAIL" in validation_result:
                corrected_sql = self.chains["sql_corrector"].invoke({
                    "original_sql": generated_sql,
                    "validation_issues": validation_result,
                    "schema_info": schema_info
                })
                final_sql = corrected_sql
            else:
                final_sql = generated_sql
            
            # Extract clean SQL from the response
            clean_sql = extract_sql_from_response(final_sql, [])
            
            logger.info(f"LangChain SQL generation completed successfully")
            
            return {
                "sql": clean_sql,
                "llm_sql": clean_sql,
                "relevant_tables": relevant_tables,
                "selected_columns": selected_columns,
                "validation_result": validation_result,
                "success": True,
                "method": "langchain"
            }
            
        except Exception as e:
            logger.error(f"LangChain SQL generation failed: {e}")
            return {
                "sql": f"-- ERROR: LangChain generation failed: {str(e)}",
                "llm_sql": f"-- ERROR: LangChain generation failed: {str(e)}",
                "success": False,
                "error": str(e),
                "method": "langchain"
            }

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

# Update the main function to use semantic approach
def format_sql_for_bi_publisher(sql: str) -> str:
    """
    Format SQL query for clean display in BI Publisher
    """
    if not sql or not sql.strip():
        return ""
    
    # Clean up the SQL
    sql = sql.strip()
    
    # Remove any leading/trailing quotes or backticks
    sql = sql.strip('"\'`')
    
    # Basic formatting - ensure proper spacing around keywords
    sql = sql.replace('SELECT', 'SELECT')
    sql = sql.replace('FROM', '\nFROM')
    sql = sql.replace('WHERE', '\nWHERE')
    sql = sql.replace('GROUP BY', '\nGROUP BY')
    sql = sql.replace('ORDER BY', '\nORDER BY')
    sql = sql.replace('HAVING', '\nHAVING')
    sql = sql.replace('JOIN', '\nJOIN')
    sql = sql.replace('LEFT JOIN', '\nLEFT JOIN')
    sql = sql.replace('RIGHT JOIN', '\nRIGHT JOIN')
    sql = sql.replace('INNER JOIN', '\nINNER JOIN')
    sql = sql.replace('OUTER JOIN', '\nOUTER JOIN')
    sql = sql.replace('UNION', '\nUNION')
    sql = sql.replace('UNION ALL', '\nUNION ALL')
    
    # Clean up multiple newlines
    import re
    sql = re.sub(r'\n\s*\n', '\n', sql)
    
    # Ensure proper indentation for sub-clauses
    lines = sql.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Decrease indent for FROM, WHERE, GROUP BY, etc.
        if line.upper().startswith(('FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING')):
            indent_level = 0
        elif line.upper().startswith(('JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN')):
            indent_level = 0
        elif line.upper().startswith(('UNION', 'UNION ALL')):
            indent_level = 0
        elif line.upper().startswith('SELECT'):
            indent_level = 0
        else:
            # Increase indent for continuation lines
            indent_level = 1
            
        # Add proper indentation
        if indent_level > 0:
            formatted_lines.append('    ' * indent_level + line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def generate_sql_from_text(user_query: str, chroma_k: int = RETRIEVE_K, model: str = None) -> Dict[str,Any]:
    """
    Main function with option to use semantic approach
    """
    # Use semantic approach for better results
    return generate_sql_from_text_semantic(user_query, chroma_k, model)

# CLI test
if __name__ == "__main__":
    import argparse, sys
    import numpy as np
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--model", default="gpt-4o-mini" if openai_client else "llama3:8b")
    p.add_argument("--method", choices=["semantic", "langchain"], default="semantic", 
                   help="Choose between semantic approach or LangChain chains approach")
    args = p.parse_args()
    
    if args.method == "langchain":
        if not LANGCHAIN_AVAILABLE:
            print("ERROR: LangChain packages not available. Install with: pip install langchain langchain-community langchain-openai langchain-ollama")
            sys.exit(1)
        try:
            sql_chain = OracleSQLChain(model_name=args.model)
            out = sql_chain.generate_sql_with_chromadb(args.query)
        except Exception as e:
            out = {"error": f"LangChain generation failed: {str(e)}", "success": False}
    else:
        out = generate_sql_from_text(args.query, model=args.model)
    
    # Add formatted SQL to output
    if out.get("success") and out.get("sql"):
        out["formatted_sql"] = format_sql_for_bi_publisher(out["sql"])
    
    # Print the JSON output
    print(json.dumps(out, indent=2, ensure_ascii=False, cls=NumpyEncoder))
    
    # Also print the clean SQL separately for easy copying
    if out.get("success") and out.get("formatted_sql"):
        print("\n" + "="*80)
        print("🎯 CLEAN SQL FOR BI PUBLISHER:")
        print("="*80)
        print(out["formatted_sql"])
        print("="*80)
        print("✅ Copy the SQL above and paste it directly into BI Publisher")
        print("="*80)