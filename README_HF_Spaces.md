# Oracle SQL Assistant - Hugging Face Spaces

## 🚀 Deployment to Hugging Face Spaces

This version uses **Gradio** instead of Streamlit for better performance on Hugging Face Spaces with **16GB RAM**.

## 📁 Files for HF Spaces

- `app_gradio.py` - Main Gradio application
- `requirements_hf.txt` - Python dependencies
- `sqlgen.py` - SQL generation logic (unchanged)
- `sqlgen_pinecone.py` - Pinecone integration (unchanged)
- `excel_generator.py` - Excel generation (unchanged)
- `process_docs/` - Process documentation (unchanged)

## 🔧 Setup Instructions

### 1. Create Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Choose **Gradio** as SDK
3. Set visibility to **Public**
4. Create the space

### 2. Upload Files

Upload these files to your HF Space:
- `app_gradio.py` → `app.py` (rename for HF Spaces)
- `requirements_hf.txt` → `requirements.txt`
- All other files from your project

### 3. Configure Secrets

In your HF Space settings, add these secrets:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
OPENAI_API_KEY=your_openai_api_key
```

### 4. Deploy

The space will automatically build and deploy. You'll get a URL like:
`https://huggingface.co/spaces/yourusername/sql-generation-app`

## 🎯 Key Differences from Streamlit Version

### UI Changes:
- **Gradio interface** instead of Streamlit
- **Different styling** but same functionality
- **Better performance** on HF Spaces

### Backend (Unchanged):
- ✅ **Same SQL generation logic**
- ✅ **Same Pinecone integration**
- ✅ **Same Excel generation**
- ✅ **Same caching mechanisms**

## 📊 Expected Performance

| Platform | RAM | Expected Performance |
|----------|-----|---------------------|
| **Streamlit Cloud** | ~1GB | 60-120s per query |
| **Hugging Face Spaces** | 16GB | 20-40s per query |

## 🎨 Gradio UI Features

- **Query input** with examples
- **Model selection** (gpt-4o-mini, gpt-4o, llama3:8b)
- **Real-time status** updates
- **SQL code display** with syntax highlighting
- **Excel export** functionality
- **Generation time** tracking
- **Responsive design**

## 🔍 Testing the Deployment

Once deployed, test with these queries:

1. **Simple**: "List all unpaid invoices due in August 2025"
2. **Medium**: "Show total receipts per customer in Q2 2025"
3. **Complex**: "For each ledger, show the last journal entry posted and its total debit amount"

## 🚨 Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check `requirements.txt` has all dependencies
   - Ensure Python 3.9+ compatibility

2. **App Crashes**:
   - Verify all secrets are set correctly
   - Check HF Spaces logs for errors

3. **Slow Performance**:
   - Monitor HF Spaces metrics
   - Consider using faster model (gpt-4o-mini)

### Monitoring:
- **HF Spaces Logs**: Real-time application logs
- **Metrics**: CPU, Memory usage
- **Community**: Share and get feedback

## 🎉 Benefits of HF Spaces

- ✅ **16GB RAM** (vs 1GB on Streamlit Cloud)
- ✅ **No sleep mode** (always responsive)
- ✅ **Better performance** for ML apps
- ✅ **Community sharing** and feedback
- ✅ **Free hosting** with good resources
- ✅ **Easy deployment** from GitHub

## 📈 Migration Checklist

- [ ] Create HF Space with Gradio SDK
- [ ] Upload `app_gradio.py` as `app.py`
- [ ] Upload `requirements_hf.txt` as `requirements.txt`
- [ ] Upload all other project files
- [ ] Set environment variables/secrets
- [ ] Test deployment
- [ ] Verify performance improvements
- [ ] Share with community

Your SQL generation app should perform much better on Hugging Face Spaces! 🚀
