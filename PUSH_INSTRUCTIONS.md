# Push to Hugging Face - Step by Step

## Step 1: Open Terminal in Project Directory
```bash
cd "c:\VASAVI COLLEGE\SEM-6\NNDL\Project\fedfusionnet_plus_plus"
```

## Step 2: Initialize Git (if not already done)
```bash
git init
```

## Step 3: Add All Files
```bash
git add .
```

## Step 4: Commit
```bash
git commit -m "Deploy FedFusionNet++ with fixed architecture - timm 1.0.26, CrossViT 240x240"
```

## Step 5: Add Hugging Face Remote
Replace YOUR_USERNAME with your HF username:
```bash
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus
```

## Step 6: Push to Hugging Face
```bash
git push origin main
```

If it asks for credentials:
- Username: Your HF username
- Password: Your HF Access Token (get from https://huggingface.co/settings/tokens)

## Step 7: Set Environment Variables in HF Space
1. Go to your Space on Hugging Face
2. Click Settings → Variables
3. Add:
   - `SECRET_KEY` = any-random-string-here
   - `GEMINI_API_KEY` = your-gemini-key (optional)
   - `MONGODB_URI` = your-mongodb-uri (optional)

## Step 8: Wait for Build
- Build takes 5-10 minutes
- Models download automatically from HF Model Hub
- Check logs for any errors

## Done!
Your app will be live at: https://huggingface.co/spaces/YOUR_USERNAME/fedfusionnet-plus
