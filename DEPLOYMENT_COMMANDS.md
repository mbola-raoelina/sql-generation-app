# 🚀 **EXACT DEPLOYMENT COMMANDS**

## **Step 1: Check Current Status**
```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
```

## **Step 2: Add Files to Git (Respecting .gitignore)**
```bash
# Add all files (large files will be automatically excluded by .gitignore)
git add .

# Check what will be committed (should NOT include large files)
git status
```

## **Step 3: Make Initial Commit**
```bash
git commit -m "Initial commit: SQL Generation App for Streamlit Cloud deployment"
```

## **Step 4: Create GitHub Repository**
1. Go to https://github.com/new
2. Repository name: `sql-generation-app` (or your choice)
3. Description: `AI-powered SQL generation with Excel export via BI Publisher`
4. Set to **Public** (required for Streamlit Cloud free tier)
5. **DO NOT** check "Add a README file", "Add .gitignore", or "Choose a license"
6. Click "Create repository"

## **Step 5: Connect to GitHub and Push**
```bash
# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/sql-generation-app.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

## **Step 6: Set Up Pinecone (Before Streamlit Deployment)**
1. Go to https://app.pinecone.io/
2. Create free account
3. Create index:
   - Name: `sqlgen-schema-docs`
   - Dimensions: `1024`
   - Metric: `cosine`
4. Get your API key

## **Step 7: Migrate Data to Pinecone**
```bash
# Install Pinecone client
pip install pinecone-client

# Run setup (will ask for your Pinecone API key)
python setup_pinecone.py

# Run migration (moves ChromaDB data to Pinecone)
python migrate_to_pinecone.py
```

## **Step 8: Deploy to Streamlit Cloud**
1. Go to https://share.streamlit.io/
2. Click "New app"
3. Connect GitHub account
4. Select repository: `YOUR_USERNAME/sql-generation-app`
5. Main file: `streamlit_hybrid_app.py`
6. Add environment variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=sqlgen-schema-docs
   PINECONE_ENVIRONMENT=us-east-1-aws
   OPENAI_API_KEY=your_openai_api_key
   BIP_USERNAME=your_bip_username
   BIP_PASSWORD=your_bip_password
   BIP_ENDPOINT=your_bip_endpoint
   BIP_WSDL_URL=your_bip_wsdl_url
   BIP_REPORT_PATH=your_report_path
   ```
7. Click "Deploy!"

---

## **📊 Expected Results:**

### **Files Pushed to GitHub:**
- ✅ `streamlit_hybrid_app.py`
- ✅ `sqlgen.py`
- ✅ `sqlgen_integrated.py`
- ✅ `sqlgen_pinecone.py`
- ✅ `excel_generator.py`
- ✅ `requirements.txt`
- ✅ `process_docs/` folder
- ✅ `README.md`
- ✅ `.streamlit/config.toml`
- ✅ `.gitignore`

### **Files Excluded (via .gitignore):**
- ❌ `schema.sql` (18.5 MB)
- ❌ `chroma_db/` (572+ MB)
- ❌ `schema_docs.jsonl` (50 MB)
- ❌ `app_env/` (virtual environment)
- ❌ `.env` (API keys)
- ❌ `*.log` (debug logs)
- ❌ `archive_*/` (old files)
- ❌ `test_*.py` (test files)
- ❌ Setup/migration scripts

### **Total Repository Size:** ~3-4 MB (well within GitHub limits)

---

## **🧪 Test Commands:**

### **Before Deployment:**
```bash
# Test Pinecone connection
python -c "from sqlgen_pinecone import test_pinecone_connection; test_pinecone_connection()"

# Test app locally
streamlit run streamlit_hybrid_app.py
```

### **After Deployment:**
1. Open your Streamlit Cloud app URL
2. Test with: "List all unreconciled bank transactions"
3. Verify Excel export works
4. Check that validation catches invalid columns

---

## **🔄 Update Commands:**
```bash
# Make changes to your code
# ...

# Commit and push updates
git add .
git commit -m "Update: [describe changes]"
git push origin main

# Streamlit Cloud auto-redeploys
```

---

## **🎯 Your App URL:**
After deployment: `https://YOUR_USERNAME-sql-generation-app.streamlit.app/`
