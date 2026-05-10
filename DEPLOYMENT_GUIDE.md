# 🚀 Hugging Face Spaces Deployment Guide

## Step-by-Step Deployment Instructions

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up for a free account

### 2. Create New Space
- Go to https://huggingface.co/new-space
- **Space name:** `fedfusionnet-plus-plus` (or your preferred name)
- **License:** MIT
- **Select SDK:** Docker
- **Space hardware:** CPU basic (free tier)
- Click **Create Space**

### 3. Get API Keys (FREE)

#### Gemini API Key (Primary VLM)
1. Go to https://makersuite.google.com/app/apikey
2. Click **Create API Key**
3. Copy the key (starts with `AIza...`)
4. **Rate Limit:** 1500 requests/day FREE

#### Groq API Key (Fallback VLM)
1. Go to https://console.groq.com/
2. Sign up/Login
3. Go to **API Keys** section
4. Click **Create API Key**
5. Copy the key (starts with `gsk_...`)
6. **Rate Limit:** Unlimited FREE

### 4. Configure Space Secrets
In your Hugging Face Space settings:

1. Click **Settings** tab
2. Scroll to **Repository secrets**
3. Add the following secrets:

```
GEMINI_API_KEY = your_gemini_api_key_here
GROQ_API_KEY = your_groq_api_key_here
FLASK_SECRET_KEY = generate_random_string_here
```

To generate FLASK_SECRET_KEY:
```python
import secrets
print(secrets.token_hex(32))
```

### 5. Push Code to Hugging Face

#### Option A: Using Git (Recommended)

```bash
# Navigate to your project directory
cd "c:\VASAVI COLLEGE\SEM-6\NNDL\Project\fedfusionnet_plus_plus"

# Add Hugging Face remote
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus-plus

# Push to Hugging Face
git push huggingface main
```

#### Option B: Using Hugging Face Hub CLI

```bash
# Install Hugging Face Hub
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Upload repository
huggingface-cli upload YOUR_USERNAME/fedfusionnet-plus-plus . --repo-type=space
```

### 6. Wait for Build
- Hugging Face will automatically build your Docker image
- This may take 10-15 minutes
- Monitor build logs in the **Logs** tab

### 7. Access Your Deployed App
- Once build completes, your app will be available at:
  ```
  https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus-plus
  ```

---

## 📋 Pre-Deployment Checklist

### Required Files (Already Created)
- ✅ `Dockerfile` - Docker configuration
- ✅ `.dockerignore` - Exclude unnecessary files
- ✅ `README_HF.md` - Hugging Face Space card
- ✅ `.env.example` - Environment variable template
- ✅ `requirements.txt` - Python dependencies

### Model Files Required
Ensure these files exist in `backend/models/`:
- ✅ `hetfusionnet_v2_FINAL.pth` (305.75 MB) - CNN model
- ✅ `stage2_tabular_model.pkl` (~500 KB) - Tabular model

**Note:** If model files are too large (>5GB), you may need to use Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Track large files with Git LFS"
```

---

## 🔧 Configuration

### Environment Variables
Set these in Hugging Face Space Settings → Repository secrets:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini 2.5 Flash API key |
| `GROQ_API_KEY` | Yes | Groq Llama 3.3 70B API key |
| `FLASK_SECRET_KEY` | Yes | Random secret key for Flask sessions |
| `MONGODB_URI` | No | MongoDB connection string (optional) |
| `PORT` | No | Port number (default: 7860) |
| `HOST` | No | Host address (default: 0.0.0.0) |

### Demo Credentials
For testing without MongoDB:
- **Email:** `demo@hospital.com`
- **Password:** `demo123`

---

## 🐛 Troubleshooting

### Build Fails
1. Check **Logs** tab for error messages
2. Verify all dependencies in `requirements.txt`
3. Ensure model files are present
4. Check Docker syntax in `Dockerfile`

### App Doesn't Start
1. Verify port 7860 is exposed in Dockerfile
2. Check environment variables are set correctly
3. Review application logs in **Logs** tab

### Models Not Loading
1. Verify model files exist in `backend/models/`
2. Check file paths in `app.py`
3. Ensure model files are tracked by Git LFS if >100MB

### VLM Not Working
1. Verify API keys are set in Space secrets
2. Check API key validity at provider websites
3. Review logs for API error messages

### MongoDB Connection Issues
1. MongoDB is optional - app works without it
2. If using MongoDB, verify `MONGODB_URI` is correct
3. Check MongoDB Atlas network access settings

---

## 📊 Resource Requirements

### Minimum Requirements (Free Tier)
- **CPU:** 2 cores
- **RAM:** 16 GB
- **Storage:** 50 GB
- **Bandwidth:** Unlimited

### Recommended for Production
- **CPU:** 4 cores
- **RAM:** 32 GB
- **Storage:** 100 GB
- **GPU:** Optional (for faster inference)

---

## 🔄 Updating Deployment

### Update Code
```bash
# Make changes locally
git add .
git commit -m "Update description"

# Push to Hugging Face
git push huggingface main
```

### Update Secrets
1. Go to Space Settings
2. Update secret values
3. Restart Space (automatic after secret change)

---

## 📈 Monitoring

### Check Application Health
```bash
curl https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus-plus/api/health
```

Expected response:
```json
{
  "status": "running",
  "mongodb": "connected",
  "models": {
    "cnn_model": "loaded",
    "tabular_model": "loaded"
  },
  "message": "FedFusionNet++ API is running"
}
```

### View Logs
- Go to Space → **Logs** tab
- Monitor real-time application logs
- Check for errors or warnings

---

## 🔒 Security Best Practices

1. **Never commit API keys** to Git repository
2. **Use Space secrets** for sensitive data
3. **Rotate API keys** regularly
4. **Enable 2FA** on Hugging Face account
5. **Monitor API usage** to detect anomalies

---

## 💰 Cost Considerations

### Free Tier Includes:
- ✅ Unlimited public Spaces
- ✅ CPU basic hardware (free)
- ✅ 50 GB storage
- ✅ Unlimited bandwidth
- ✅ Community support

### Paid Upgrades (Optional):
- 🚀 GPU hardware ($0.60/hour)
- 🚀 Persistent storage (additional)
- 🚀 Private Spaces
- 🚀 Priority support

**For this project:** Free tier is sufficient for testing and moderate usage.

---

## 📞 Support

### Hugging Face Support
- **Documentation:** https://huggingface.co/docs/hub/spaces
- **Forum:** https://discuss.huggingface.co/
- **Discord:** https://discord.gg/hugging-face

### Project Support
- **GitHub Issues:** https://github.com/afridpasha/FedFusionNet-/issues
- **Email:** Contact repository owner

---

## ✅ Post-Deployment Checklist

After successful deployment:

- [ ] Test login with demo credentials
- [ ] Upload test histopathology image
- [ ] Verify CNN prediction works
- [ ] Check XAI heatmaps generate
- [ ] Test VLM clinical narrative
- [ ] Verify WSI spatial analysis
- [ ] Test ChatBot Q&A
- [ ] Check multi-language translation
- [ ] Test PDF report generation
- [ ] Verify responsive design on mobile

---

## 🎉 Success!

Your FedFusionNet++ application is now deployed on Hugging Face Spaces!

**Share your Space:**
```
https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus-plus
```

**Embed in website:**
```html
<iframe
  src="https://YOUR_USERNAME-fedfusionnet-plus-plus.hf.space"
  frameborder="0"
  width="850"
  height="450"
></iframe>
```

---

**Made with ❤️ for advancing medical AI**
