# Oracle Fusion Financials SQL Generation System

## 🎯 **PROJECT OVERVIEW**

This is an intelligent SQL generation system for Oracle Fusion Financials that uses semantic understanding and vector embeddings to automatically generate SQL queries from natural language. The system combines multiple approaches including ChromaDB vector search, business process documentation, and OpenAI's GPT models to create accurate, context-aware SQL queries.

## 🏗️ **SYSTEM ARCHITECTURE**

### **Core Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  SQL Generator  │────│   Excel Export  │
│  (User Input)   │    │   (Core Logic)  │    │ (BI Publisher)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   ChromaDB      │              │
         └──────────────│  (Vector Store) │──────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  OpenAI API     │
                        │ (GPT-4o-mini)   │
                        └─────────────────┘
```

### **Data Flow**

1. **User Input** → Natural language query via Streamlit UI
2. **Semantic Analysis** → Vector embedding of query + business context
3. **Table Selection** → ChromaDB search for relevant tables/columns
4. **SQL Generation** → OpenAI GPT creates SQL with business context
5. **Validation** → Schema validation and error checking
6. **Export** → BI Publisher integration for Excel generation

## 📁 **PROJECT STRUCTURE**

```
app_sql/
├── 📄 Core Application Files
│   ├── streamlit_hybrid_app.py          # Main Streamlit UI
│   ├── sqlgen_integrated.py             # Bridge between UI and core logic
│   ├── sqlgen.py                        # Core SQL generation engine
│   └── excel_generator.py               # BI Publisher Excel export
│
├── 📁 Data & Configuration
│   ├── .env                             # Environment variables
│   ├── schema.sql                       # Oracle Fusion schema (19MB)
│   ├── schema_docs.jsonl                # Processed schema documentation
│   └── requirements.txt                 # Python dependencies
│
├── 📁 Vector Database
│   └── chroma_db/                       # ChromaDB persistent storage
│       ├── chroma.sqlite3               # Vector database
│       └── [embedding files]            # Cached embeddings
│
├── 📁 Business Process Documentation
│   └── process_docs/                    # Business process context
│       ├── general_ledger.txt           # GL process documentation
│       ├── fixed_asset.txt              # FA process documentation
│       ├── order_to_cash.txt            # O2C process documentation
│       └── procure_to_pay.txt           # P2P process documentation
│
├── 📁 Utilities & Setup
│   ├── ingest_schema_to_chroma.py       # Schema ingestion script
│   ├── integrate_process_docs.py        # Process docs integration
│   └── test_validation_fix.py           # Validation testing
│
└── 📁 Archive (after cleanup)
    ├── archive_old_files/               # Deprecated files
    ├── archive_test_files/              # Test scripts
    ├── archive_debug_logs/              # Debug logs
    └── archive_duplicate_files/         # Duplicate files
```

## ⚙️ **SYSTEM CONFIGURATION**

### **Environment Variables (.env)**

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# BI Publisher Configuration
BIP_USERNAME=your_bip_username
BIP_PASSWORD=your_bip_password
BIP_WSDL_URL=https://your-bip-server/xmlpserver/services/v2/ReportService?wsdl
BIP_ENDPOINT=https://your-bip-server

# Report Paths (URL encoded)
BIP_REPORT_PATH_GL=/xdo/ouput/GL%20Reports/GL_Journal_Entries_Report.xdo
BIP_REPORT_PATH_AR=/xdo/ouput/AR%20Reports/AR_Invoice_Report.xdo
BIP_REPORT_PATH_AP=/xdo/ouput/AP%20Reports/AP_Invoice_Report.xdo
```

### **Dependencies (requirements.txt)**

```
streamlit>=1.28.0
openai>=1.3.0
chromadb>=0.4.15
sentence-transformers>=2.2.2
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
zeep>=4.2.1
selenium>=4.15.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

## 🚀 **INSTALLATION & SETUP**

### **1. Environment Setup**

```bash
# Create virtual environment
python -m venv app_env
app_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Data Preparation**

```bash
# Ingest Oracle Fusion schema into ChromaDB
python ingest_schema_to_chroma.py

# Integrate business process documentation
python integrate_process_docs.py
```

### **3. Configuration**

```bash
# Copy and configure environment variables
copy .env.example .env
# Edit .env with your actual credentials
```

## 🎮 **USAGE COMMANDS**

### **Start the Application**

```bash
# Activate virtual environment
app_env\Scripts\activate

# Launch Streamlit application
streamlit run streamlit_hybrid_app.py
```

### **Run Tests**

```bash
# Test validation fixes
python test_validation_fix.py

# Test comprehensive queries
python test_comprehensive_queries.py
```

### **Development Commands**

```bash
# Re-ingest schema (if schema.sql updated)
python ingest_schema_to_chroma.py

# Update process documentation
python integrate_process_docs.py

# Clean project (run cleanup command above)
```

## 🧠 **CORE ALGORITHMS**

### **1. Semantic Table Selection**

```python
def get_related_tables(query_embedding, top_k=5):
    """
    Uses ChromaDB vector search to find semantically relevant tables
    based on table comments and business process context
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"document_type": "table"}
    )
    return results['metadatas'][0]
```

### **2. Dynamic Primary Table Detection**

```python
def get_dynamic_primary_tables():
    """
    Extracts primary transactional tables from process_docs
    Replaces hardcoded PRIMARY_TABLE_INDICATORS with business-driven detection
    """
    # Reads process_docs/*.txt files
    # Extracts table names using regex patterns
    # Returns sorted list of primary tables
```

### **3. Business Process Context Integration**

```python
def get_process_context_for_query(user_query):
    """
    Retrieves relevant business process context using ChromaDB
    Enhances SQL generation with business logic understanding
    """
    query_embedding = model.encode([user_query])
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"document_type": "business_process"}
    )
    return format_process_context(results)
```

### **4. SQL Generation with Validation**

```python
def generate_sql_from_text_semantic(user_query, model="gpt-4o-mini"):
    """
    Generates SQL using OpenAI GPT with business context
    Includes comprehensive validation and error handling
    """
    # 1. Get semantic tables
    # 2. Get business process context  
    # 3. Build enhanced prompt
    # 4. Generate SQL with GPT
    # 5. Extract and validate SQL
    # 6. Return formatted result
```

## 🔧 **KEY FEATURES**

### **Semantic Understanding**
- **Vector Embeddings**: Uses SentenceTransformer for semantic matching
- **Business Context**: Integrates process documentation for better accuracy
- **Dynamic Selection**: No hardcoded table/column lists

### **Intelligent SQL Generation**
- **Context-Aware**: Uses business process knowledge
- **Schema Validation**: Comprehensive error checking
- **Multiple Methods**: Semantic, LangChain, and hybrid approaches

### **Excel Export Integration**
- **BI Publisher**: Direct SOAP API integration
- **Selenium Fallback**: UI automation for complex reports
- **Simple Fallback**: CSV export when BI Publisher unavailable

### **Performance Optimizations**
- **Model Caching**: Persistent embedding model storage
- **Vector Caching**: ChromaDB persistent storage
- **Process Context Caching**: Query result caching
- **Batch Operations**: Optimized database queries

## 📊 **VALIDATION SYSTEM**

### **Dynamic Validation**

```python
def validate_sql_against_schema(sql, available_columns):
    """
    Validates generated SQL against actual schema
    Uses dynamic patterns instead of hardcoded lists
    """
    # 1. Extract column references from SQL
    # 2. Map table aliases to actual table names
    # 3. Validate columns exist in schema
    # 4. Provide suggestions for errors
```

### **Error Handling**

- **Column Validation**: Checks if columns exist in schema
- **Table Validation**: Verifies table names and relationships
- **Join Validation**: Validates foreign key relationships
- **Syntax Validation**: Basic SQL syntax checking

## 🔄 **SYSTEM WORKFLOW**

### **Query Processing Pipeline**

```
User Query → Embedding → ChromaDB Search → Table Selection
     ↓
Business Context → Enhanced Prompt → GPT Generation
     ↓
SQL Extraction → Validation → Error Handling → Result
     ↓
Excel Export (Optional) → BI Publisher → Download
```

### **Data Sources**

1. **Oracle Fusion Schema** (schema.sql)
   - 19MB of table/column definitions
   - Foreign key relationships
   - Data types and constraints

2. **Business Process Documentation** (process_docs/)
   - General Ledger processes
   - Fixed Assets procedures
   - Order-to-Cash workflows
   - Procure-to-Pay processes

3. **Vector Embeddings** (chroma_db/)
   - Cached semantic representations
   - Persistent vector storage
   - Fast similarity searches

## 🐛 **TROUBLESHOOTING**

### **Common Issues**

1. **Empty SQL Generation**
   - Check OpenAI API key in .env
   - Verify ChromaDB has data
   - Check debug logs for errors

2. **Validation Failures**
   - Ensure schema ingestion completed
   - Check table alias mapping
   - Verify column names in schema

3. **Excel Export Issues**
   - Verify BI Publisher credentials
   - Check network connectivity
   - Review SOAP endpoint configuration

### **Debug Logs**

```bash
# Check recent debug logs
dir sqlgen_debug_*.log
dir excel_debug_*.log

# View log contents
type sqlgen_debug_YYYYMMDD_HHMMSS.log
```

## 🔮 **FUTURE ENHANCEMENTS**

### **Planned Improvements**

1. **Fully Dynamic Table Selection**
   - Replace all hardcoded patterns
   - Business process-driven detection
   - Machine learning-based ranking

2. **Enhanced Semantic Matching**
   - Multi-modal embeddings
   - Context-aware similarity
   - Performance optimization

3. **Advanced SQL Features**
   - Complex aggregations
   - Window functions
   - Advanced joins

4. **Real-time Integration**
   - Live schema updates
   - Dynamic process documentation
   - Continuous learning

## 📞 **SUPPORT & CONTINUATION**

### **For New Chat Sessions**

This README provides complete context for continuing development. Key points:

1. **Current Status**: System is functional with semantic SQL generation
2. **Main Issues**: Validation logic needs refinement for table aliases
3. **Architecture**: ChromaDB + OpenAI + Streamlit + BI Publisher
4. **Data Sources**: Oracle Fusion schema + business process docs
5. **Performance**: Optimized with caching and vector storage

### **Quick Start for New Session**

```bash
# 1. Activate environment
app_env\Scripts\activate

# 2. Check system status
python test_validation_fix.py

# 3. Launch application
streamlit run streamlit_hybrid_app.py
```

---

**Last Updated**: September 17, 2025  
**Version**: 2.0 (Hybrid Semantic System)  
**Status**: Functional with validation improvements needed