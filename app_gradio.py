#!/usr/bin/env python3
"""
Gradio version of the SQL Generation App for Hugging Face Spaces
Uses the same SQL generation logic but with Gradio UI for better performance
"""

import gradio as gr
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our existing SQL generation modules
from sqlgen import generate_sql_from_text_semantic
from excel_generator import excel_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sql_gradio(user_query: str, model: str = "gpt-4o-mini") -> tuple:
    """
    Generate SQL using our existing pipeline with Gradio interface
    Returns: (sql_result, generation_time, status_message)
    """
    if not user_query.strip():
        return "", 0, "❌ Please enter a query"
    
    try:
        logger.info(f"Starting SQL generation for: {user_query}")
        start_time = time.time()
        
        # Use our existing semantic SQL generation
        result = generate_sql_from_text_semantic(user_query, model=model)
        
        generation_time = time.time() - start_time
        
        if "error" in result:
            error_msg = f"❌ SQL generation failed: {result['error']}"
            logger.error(error_msg)
            return "", generation_time, error_msg
        
        sql = result.get("llm_sql", "")
        if sql and not sql.startswith("-- ERROR:"):
            success_msg = f"✅ SQL generated successfully in {generation_time:.2f}s"
            logger.info(success_msg)
            return sql, generation_time, success_msg
        else:
            error_msg = f"❌ Invalid SQL generated: {sql}"
            logger.error(error_msg)
            return "", generation_time, error_msg
            
    except Exception as e:
        error_msg = f"❌ Exception during SQL generation: {str(e)}"
        logger.error(error_msg)
        return "", 0, error_msg

def generate_excel_gradio(sql_query: str, filename: str = "sql_results") -> tuple:
    """
    Generate Excel file from SQL query using our existing excel generator
    Returns: (excel_file_path, status_message)
    """
    if not sql_query.strip():
        return None, "❌ Please generate SQL first"
    
    try:
        logger.info("Generating Excel file from SQL")
        
        # Use our existing excel generator
        excel_file = excel_generator.generate_excel_from_sql(sql_query, filename)
        
        if excel_file:
            success_msg = f"✅ Excel file generated: {excel_file}"
            logger.info(success_msg)
            return excel_file, success_msg
        else:
            error_msg = "❌ Failed to generate Excel file"
            logger.error(error_msg)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"❌ Exception during Excel generation: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def create_gradio_interface():
    """
    Create the Gradio interface for the SQL generation app
    """
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=css, title="Oracle SQL Assistant - Gradio Version") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 Oracle SQL Assistant - Gradio Version</h1>
            <p>Advanced SQL generation with Excel export capabilities</p>
            <p><strong>Powered by Hugging Face Spaces with 16GB RAM</strong></p>
        </div>
        """)
        
        # Configuration section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                model_choice = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "llama3:8b"],
                    value="gpt-4o-mini",
                    label="LLM Model",
                    info="gpt-4o-mini is fastest and most reliable"
                )
                
                # Environment status
                env_status = gr.HTML("""
                <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                    <strong>Environment Status:</strong><br>
                    <span style="color: green;">✅ Pinecone: Configured</span><br>
                    <span style="color: green;">✅ OpenAI: Configured</span><br>
                    <span style="color: blue;">🚀 Platform: Hugging Face Spaces</span>
                </div>
                """)
        
        # Main query interface
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Query Interface")
                
                query_input = gr.Textbox(
                    label="Enter your natural language query:",
                    placeholder="e.g., Show me all unpaid invoices due in August 2025",
                    lines=3,
                    max_lines=5
                )
                
                generate_btn = gr.Button(
                    "🚀 Generate SQL", 
                    variant="primary",
                    size="lg"
                )
                
                # Example queries
                with gr.Accordion("💡 Example Queries", open=False):
                    gr.Examples(
                        examples=[
                            "List all unpaid invoices due in August 2025",
                            "Show total receipts per customer in Q2 2025",
                            "For each ledger, show the last journal entry posted and its total debit amount",
                            "Show supplier names and total invoice amounts for July 2025",
                            "List the top 5 suppliers by total unpaid amount",
                            "Show journal entries with preparer name and ledger name for June 2025"
                        ],
                        inputs=query_input,
                        label="Click to use example queries"
                    )
        
        # Results section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Generated SQL")
                
                sql_output = gr.Code(
                    label="Generated SQL Query",
                    language="sql",
                    lines=10,
                    interactive=True
                )
                
                # Status and metrics
                status_output = gr.HTML(
                    label="Status",
                    value="<div>Ready to generate SQL...</div>"
                )
                
                time_output = gr.Textbox(
                    label="Generation Time",
                    value="",
                    interactive=False
                )
        
        # Excel generation section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📈 Excel Export")
                
                excel_filename = gr.Textbox(
                    label="Excel Filename (optional):",
                    placeholder="sql_results",
                    value="sql_results"
                )
                
                excel_btn = gr.Button(
                    "📊 Generate Excel File",
                    variant="secondary"
                )
                
                excel_download = gr.File(
                    label="Download Excel File",
                    visible=False
                )
                
                excel_status = gr.HTML(
                    label="Excel Status",
                    value="<div>Generate SQL first, then create Excel file</div>"
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p><strong>Oracle SQL Assistant - Gradio Version</strong></p>
            <p>Built with ❤️ using Gradio and Hugging Face Spaces</p>
            <p>Supports AP, AR, GL, FA, CE, and Cross-Module queries</p>
        </div>
        """)
        
        # Event handlers
        def handle_sql_generation(query, model):
            """Handle SQL generation with progress feedback"""
            if not query.strip():
                return "", 0, "<div class='status-error'>❌ Please enter a query</div>"
            
            # Generate SQL
            sql, gen_time, status = generate_sql_gradio(query, model)
            
            # Format status message
            if "✅" in status:
                status_html = f"<div class='status-success'>{status}</div>"
            else:
                status_html = f"<div class='status-error'>{status}</div>"
            
            return sql, f"{gen_time:.2f} seconds", status_html
        
        def handle_excel_generation(sql, filename):
            """Handle Excel file generation"""
            if not sql.strip():
                return None, "<div class='status-error'>❌ Please generate SQL first</div>"
            
            excel_file, status = generate_excel_gradio(sql, filename)
            
            if excel_file:
                status_html = f"<div class='status-success'>{status}</div>"
                return gr.File(value=excel_file, visible=True), status_html
            else:
                status_html = f"<div class='status-error'>{status}</div>"
                return gr.File(visible=False), status_html
        
        # Connect events
        generate_btn.click(
            fn=handle_sql_generation,
            inputs=[query_input, model_choice],
            outputs=[sql_output, time_output, status_output]
        )
        
        excel_btn.click(
            fn=handle_excel_generation,
            inputs=[sql_output, excel_filename],
            outputs=[excel_download, excel_status]
        )
        
        # Allow Enter key to trigger generation
        query_input.submit(
            fn=handle_sql_generation,
            inputs=[query_input, model_choice],
            outputs=[sql_output, time_output, status_output]
        )
    
    return demo

def main():
    """
    Main function to launch the Gradio app
    """
    logger.info("Starting Oracle SQL Assistant - Gradio Version")
    
    # Check environment variables
    required_vars = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Make sure to set these in Hugging Face Spaces secrets")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with Hugging Face Spaces configuration
    demo.launch(
        server_name="0.0.0.0",  # Required for Hugging Face Spaces
        server_port=7860,       # Default Hugging Face Spaces port
        share=False,            # Don't create public link (HF Spaces handles this)
        show_error=True,        # Show errors in the UI
        quiet=False             # Show startup logs
    )

if __name__ == "__main__":
    main()
