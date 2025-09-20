#!/usr/bin/env python3
"""
Test script for the Gradio app
Verifies that the Gradio interface works correctly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gradio_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing Gradio app imports...")
    
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gradio: {e}")
        return False
    
    try:
        from sqlgen import generate_sql_from_text_semantic
        print("✅ SQL generation module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import sqlgen: {e}")
        return False
    
    try:
        from excel_generator import excel_generator
        print("✅ Excel generator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import excel_generator: {e}")
        return False
    
    return True

def test_environment_variables():
    """Test that required environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT", 
        "PINECONE_INDEX_NAME",
        "OPENAI_API_KEY"
    ]
    
    all_set = True
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Not set")
            all_set = False
    
    return all_set

def test_gradio_interface():
    """Test that the Gradio interface can be created"""
    print("\n🔍 Testing Gradio interface creation...")
    
    try:
        from app_gradio import create_gradio_interface
        demo = create_gradio_interface()
        print("✅ Gradio interface created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create Gradio interface: {e}")
        return False

def test_sql_generation():
    """Test SQL generation functionality"""
    print("\n🔍 Testing SQL generation...")
    
    try:
        from app_gradio import generate_sql_gradio
        
        # Test with a simple query
        test_query = "List all unpaid invoices due in August 2025"
        sql, gen_time, status = generate_sql_gradio(test_query)
        
        if "✅" in status:
            print(f"✅ SQL generation successful: {status}")
            print(f"   Generation time: {gen_time:.2f}s")
            print(f"   SQL preview: {sql[:100]}...")
            return True
        else:
            print(f"❌ SQL generation failed: {status}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during SQL generation test: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Gradio App for Hugging Face Spaces")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_gradio_imports),
        ("Environment Variables", test_environment_variables),
        ("Gradio Interface", test_gradio_interface),
        ("SQL Generation", test_sql_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Ready for Hugging Face Spaces deployment.")
    else:
        print("⚠️ Some tests failed. Please fix issues before deployment.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
