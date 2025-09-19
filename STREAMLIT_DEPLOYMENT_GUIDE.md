# 🚀 Streamlit Cloud Deployment Guide

## 📋 **Pre-Deployment Checklist**

### ✅ **Files That Will Be Pushed to GitHub:**
- ✅ `streamlit_hybrid_app.py` (main app)
- ✅ `sqlgen.py` (core SQL generation)
- ✅ `sqlgen_integrated.py` (integration layer)
- ✅ `sqlgen_pinecone.py` (Pinecone integration)
- ✅ `excel_generator.py` (Excel export)
- ✅ `requirements.txt` (dependencies)
- ✅ `process_docs/` (business process documentation)
- ✅ `README.md` (documentation)
- ✅ `.streamlit/config.toml` (Streamlit config)

### ❌ **Files That Will Be Excluded (via .gitignore):**
- ❌ `schema.sql` (18.5 MB - too large)
- ❌ `chroma_db/` (572+ MB - too large)
- ❌ `schema_docs.jsonl` (50 MB - too large)
- ❌ `app_env/` (virtual environment)
- ❌ `.env` (API keys)
- ❌ `*.log` (debug logs)
- ❌ `archive_*/` (old files)
- ❌ `test_*.py` (test files)
- ❌ `setup_pinecone.py` (temporary setup)
- ❌ `migrate_to_pinecone.py` (temporary migration)
- ❌ `deploy_to_streamlit.py` (temporary deployment script)

---

## 🔧 **Step-by-Step Deployment Process**

### **Step 1: Initialize Git Repository**
```bash
# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# Make initial commit
git commit -m "Initial commit: SQL Generation App for Streamlit Cloud"
```

### **Step 2: Create GitHub Repository**
1. Go to [GitHub.com](https://github.com)
2. Click "New repository"
3. Repository name: `sql-generation-app` (or your preferred name)
4. Description: `AI-powered SQL generation with Excel export via BI Publisher`
5. Set to **Public** (required for Streamlit Cloud free tier)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click "Create repository"

### **Step 3: Push to GitHub**
```bash
# Add GitHub remote (replace YOUR_USERNAME and YOUR_REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### **Step 4: Set Up Pinecone (Free Tier)**
1. Go to [Pinecone.io](https://app.pinecone.io/)
2. Create free account (no credit card required)
3. Create new project
4. Create index:
   - **Name**: `sqlgen-schema-docs`
   - **Dimensions**: `1024` (for thenlper/gte-base model)
   - **Metric**: `cosine`
5. Get your API key from the dashboard

### **Step 5: Migrate Data to Pinecone**
```bash
# Install Pinecone client
pip install pinecone-client

# Run setup script
python setup_pinecone.py

# Run migration script
python migrate_to_pinecone.py
```

### **Step 6: Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `YOUR_USERNAME/YOUR_REPO_NAME`
5. Set main file: `streamlit_hybrid_app.py`
6. Add environment variables (see below)
7. Click "Deploy!"

---

## 🔐 **Environment Variables for Streamlit Cloud**

Add these in your Streamlit Cloud app settings:

### **Required Variables:**
```
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=sqlgen-schema-docs
PINECONE_ENVIRONMENT=us-east-1-aws
OPENAI_API_KEY=your_openai_api_key_here
```

### **BI Publisher Variables:**
```
BIP_USERNAME=your_bip_username
BIP_PASSWORD=your_bip_password
BIP_ENDPOINT=your_bip_endpoint
BIP_WSDL_URL=your_bip_wsdl_url
BIP_REPORT_PATH=your_report_path
```

---

## 📊 **Expected Repository Size**

After deployment, your GitHub repository will contain:
- **Core application files**: ~2-3 MB
- **Process documentation**: ~1 MB
- **Total repository size**: ~3-4 MB (well within GitHub limits)

**Excluded large files:**
- `schema.sql`: 18.5 MB → **Excluded**
- `chroma_db/`: 572+ MB → **Excluded** (migrated to Pinecone)
- `schema_docs.jsonl`: 50 MB → **Excluded** (migrated to Pinecone)

---

## 🧪 **Testing Your Deployment**

### **Local Testing (Before Deployment):**
```bash
# Test Pinecone connection
python -c "from sqlgen_pinecone import test_pinecone_connection; test_pinecone_connection()"

# Test the app locally
streamlit run streamlit_hybrid_app.py
```

### **Production Testing (After Deployment):**
1. Open your Streamlit Cloud app URL
2. Test with sample queries:
   - "List all unreconciled bank transactions"
   - "Show me all invoices with their payment status"
   - "Get all customers with outstanding balances"
3. Verify Excel export functionality
4. Check logs for any errors

---

## 🔄 **Updating Your Deployment**

To update your deployed app:

```bash
# Make changes to your code
# ...

# Commit changes
git add .
git commit -m "Update: [describe your changes]"

# Push to GitHub
git push origin main

# Streamlit Cloud will automatically redeploy
```

---

## 🆘 **Troubleshooting**

### **Common Issues:**

1. **"No module named 'pinecone'"**
   - Ensure `pinecone-client>=3.0.0` is in `requirements.txt`

2. **"Pinecone connection failed"**
   - Check your `PINECONE_API_KEY` environment variable
   - Verify your Pinecone index exists and has the correct name

3. **"No documents found"**
   - Run the migration script to populate Pinecone
   - Check that your Pinecone index contains data

4. **"BI Publisher connection failed"**
   - Verify all BI Publisher environment variables
   - Check your Oracle Cloud instance is accessible

### **Logs and Debugging:**
- Streamlit Cloud logs are available in the app dashboard
- Check the "Logs" tab in your Streamlit Cloud app settings

---

## 🎉 **Success!**

Once deployed, your app will be available at:
`https://YOUR_USERNAME-YOUR_REPO_NAME.streamlit.app/`

The app will:
- ✅ Use Pinecone for vector storage (no local files needed)
- ✅ Generate SQL from natural language
- ✅ Validate SQL against schema
- ✅ Export results to Excel via BI Publisher
- ✅ Handle all the validation fixes we implemented

---

## 📞 **Support**

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all environment variables are set
3. Test Pinecone connection separately
4. Review the deployment checklist above
