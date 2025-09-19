"""
Hybrid Streamlit App
Combines sqlgen.py (SQL generation) with modular_app.py (UI + Excel features)
"""

import streamlit as st
import time
import json
import os
import logging
import subprocess
import sys
from datetime import datetime
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

# Import our excel generator and integrated SQL generator
from excel_generator import excel_generator
from sqlgen_integrated import generate_sql_integrated

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_sqlgen_integrated(user_query: str, model: str = "gpt-4o-mini", method: str = "langchain") -> dict:
    """
    Call SQL generation using direct function calls (no subprocess)
    This avoids reloading the embedding model every time
    """
    try:
        logger.info(f"Generating SQL with method: {method}, model: {model}")
        
        # Use direct function call instead of subprocess
        result = generate_sql_integrated(user_query, model, method)
        
        if result["success"]:
            logger.info("SQL generation completed successfully")
            return result
        else:
            logger.error(f"SQL generation failed: {result['error']}")
            return result
            
    except Exception as e:
        logger.error(f"Error in integrated SQL generation: {e}")
        return {
            "success": False,
            "data": None,
            "error": f"Unexpected error: {e}"
        }

def validate_direct_sql(sql_query: str) -> tuple[bool, str]:
    """Validate a directly entered SQL query with basic checks."""
    try:
        logger.info(f"Validating direct SQL: {sql_query}")
        
        # Basic SQL validation
        sql_upper = sql_query.upper()
        
        # Check for basic SQL structure
        if not sql_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        if 'FROM' not in sql_upper:
            return False, "Query must contain FROM clause"
        
        logger.info("Direct SQL validation successful")
        return True, "SQL is valid"
        
    except Exception as e:
        logger.error(f"Error validating direct SQL: {e}")
        return False, f"Validation error: {e}"

def main():
    st.set_page_config(
        page_title="Oracle SQL Assistant - Hybrid", 
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Oracle SQL Assistant - Hybrid Version")
    st.markdown("**Combines advanced SQL generation with Excel export capabilities**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # LLM Provider Selection
    llm_provider = st.sidebar.selectbox(
        "Choose LLM Provider:",
        ["gpt-4o-mini", "sqlcoder:15b", "llama3:8b", "mistral:instruct"],
        index=0,
        help="Select the LLM model for SQL generation"
    )
    
    # Method Selection
    method = st.sidebar.selectbox(
        "Choose Generation Method:",
        ["semantic", "langchain"],
        index=0,
        help="Semantic: Faster semantic approach (recommended for testing)\nLangChain: Advanced validation chains (slower, first run downloads models)"
    )
    
    # Display configuration status
    if llm_provider.startswith("gpt"):
        st.sidebar.success("✅ Using OpenAI API")
    else:
        st.sidebar.info("📡 Using Ollama local models")
    
    st.sidebar.info(f"🔧 Method: {method.upper()}")
    
    # First-time setup warning
    if method == "langchain":
        st.sidebar.warning("⚠️ First run may take 5+ minutes due to model download")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mode selection
        mode = st.radio(
            "Choose your input mode:",
            ["Natural Language Query", "Direct SQL Input"],
            horizontal=True
        )
        
        if mode == "Natural Language Query":
            # Natural language input
            user_query = st.text_area(
                "Enter your query in natural language:",
                placeholder="e.g., 'Show total receipts per customer in Q2 2025'",
                height=100
            )
            
            if st.button("🚀 Generate SQL", type="primary", use_container_width=True):
                if user_query.strip():
                    # Show progress message based on method
                    if method == "langchain":
                        progress_msg = "🔄 Processing with LangChain (may take 2-10 minutes)..."
                    else:
                        progress_msg = "🔄 Processing with semantic method (may take 1-3 minutes due to embedding processing)..."
                    
                    with st.spinner(progress_msg):
                        start_time = time.time()
                        
                        # Call our integrated SQL generator
                        result = call_sqlgen_integrated(user_query.strip(), llm_provider, method)
                        
                        duration = time.time() - start_time
                        
                        if result["success"]:
                            data = result["data"]
                            
                            # Store results in session state
                            st.session_state.generated_sql = data.get("llm_sql", data.get("formatted_sql", data.get("sql", "")))
                            st.session_state.is_valid = data.get("success", True)
                            st.session_state.processing_metrics = {
                                "duration": duration,
                                "method": method,
                                "model": llm_provider,
                                "tables": data.get("relevant_tables", ""),
                                "validation": data.get("validation_result", "")
                            }
                            
                            st.success("✅ SQL generated successfully!")
                            
                        else:
                            st.error(f"❌ SQL generation failed: {result['error']}")
                            st.session_state.generated_sql = None
                            st.session_state.is_valid = False
                else:
                    st.warning("Please enter a query")
        
        else:
            # Direct SQL input
            direct_sql = st.text_area(
                "Enter your SQL query directly:",
                placeholder="SELECT b.CUSTOMER_TRX_ID, SUM(a.AMOUNT) AS TOTAL_RECEIPTS FROM AR_CASH_RECEIPTS_ALL a JOIN AR_RECEIVABLE_APPLICATIONS_ALL b ON a.CASH_RECEIPT_ID = b.CASH_RECEIPT_ID WHERE a.RECEIPT_DATE BETWEEN TO_DATE('2025-04-01', 'YYYY-MM-DD') AND TO_DATE('2025-06-30', 'YYYY-MM-DD') GROUP BY b.CUSTOMER_TRX_ID",
                height=150
            )
            
            if st.button("🔍 Validate SQL", type="primary", use_container_width=True):
                if direct_sql.strip():
                    with st.spinner("Validating SQL..."):
                        is_valid, message = validate_direct_sql(direct_sql.strip())
                        
                        st.session_state.generated_sql = direct_sql.strip()
                        st.session_state.is_valid = is_valid
                        st.session_state.processing_metrics = {
                            "duration": 0,
                            "method": "direct",
                            "model": "none",
                            "validation": message
                        }
                        
                        if is_valid:
                            st.success("✅ SQL is valid")
                        else:
                            st.error(f"❌ SQL validation failed: {message}")
                else:
                    st.warning("Please enter a SQL query")
    
    with col2:
        # Performance metrics
        if hasattr(st.session_state, 'processing_metrics'):
            st.subheader("📊 Performance Metrics")
            metrics = st.session_state.processing_metrics
            
            if 'duration' in metrics and metrics['duration'] > 0:
                st.metric("Processing Time", f"{metrics['duration']:.2f}s")
            
            if 'method' in metrics:
                st.metric("Method", metrics['method'].upper())
            
            if 'model' in metrics:
                st.metric("Model", metrics['model'])
            
            if 'validation' in metrics and metrics['validation']:
                with st.expander("Validation Details"):
                    st.text(metrics['validation'])
    
    # Display generated SQL
    if hasattr(st.session_state, 'generated_sql') and st.session_state.generated_sql:
        st.subheader("📋 Generated SQL")
        
        # Display SQL in a nice code block
        st.code(st.session_state.generated_sql, language="sql")
        
        # Copy button
        if st.button("📋 Copy SQL", use_container_width=True):
            st.code(st.session_state.generated_sql, language="sql")
            st.success("SQL copied to clipboard!")
        
        # Excel export section
        st.subheader("📊 Export to Excel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generate Excel Report", type="secondary", use_container_width=True):
                try:
                    with st.spinner("🔧 Generating Excel report..."):
                        # Use our excel generator
                        excel_data, row_count = excel_generator.handle_sql_to_excel(st.session_state.generated_sql)
                        
                        if excel_data:
                            # Download button
                            st.download_button(
                                label="📥 Download Excel File",
                                data=excel_data,
                                file_name=f"sql_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            
                            if row_count > 0:
                                st.success(f"✅ Excel file ready! Contains {row_count} rows of data.")
                            else:
                                st.warning("⚠️ Excel file generated but contains no data.")
                        else:
                            st.error("❌ Failed to generate Excel file")
                except Exception as e:
                    st.error(f"❌ Error generating Excel: {e}")
        
        with col2:
            # Additional info
            st.info("""
            **Excel Export Features:**
            - Connects to BI Publisher API
            - Executes SQL and retrieves data
            - Formats results in Excel
            - Includes proper styling
            """)
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **Natural Language Examples:**
        - "Show total receipts per customer in Q2 2025"
        - "List all unpaid invoices due in August 2025"
        - "Show supplier names and total invoice amounts for July 2025"
        - "List the top 5 suppliers by total unpaid amount"
        - "Show journal entries with preparer name and ledger name for June 2025"
        
        **Direct SQL Examples:**
        ```sql
        SELECT b.CUSTOMER_TRX_ID, SUM(a.AMOUNT) AS TOTAL_RECEIPTS
        FROM AR_CASH_RECEIPTS_ALL a
        JOIN AR_RECEIVABLE_APPLICATIONS_ALL b ON a.CASH_RECEIPT_ID = b.CASH_RECEIPT_ID
        WHERE a.RECEIPT_DATE BETWEEN TO_DATE('2025-04-01', 'YYYY-MM-DD') AND TO_DATE('2025-06-30', 'YYYY-MM-DD')
        GROUP BY b.CUSTOMER_TRX_ID;
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Oracle SQL Assistant - Hybrid Version | Powered by sqlgen.py + Excel Generator</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
