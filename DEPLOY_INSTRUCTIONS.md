# 🚀 Deploy to Render - Instructions

Your deployment folder is ready with all required files!

## 📦 What's Included

✅ API code (4 files in `api/`)
✅ Utilities (5 files in `utils/`)
✅ Model file (`models/sign_gru_model.h5` - 2.61 MB)
✅ Data files (2 files in `data/processed/`)
✅ Configuration files (`requirements_api.txt`, `render.yaml`, `.gitignore`)

**Total: 35 files, ~10-15 MB**

## 🚀 Next Steps

### Step 1: Initialize Git
```bash
git init
git add .
git commit -m "Initial deployment to Render"
```

### Step 2: Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository (e.g., "sign-language-api")
3. Don't initialize with README (we already have files)

### Step 3: Push to GitHub
```bash
# Replace YOUR_USERNAME and YOUR_REPO with your actual values
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Render
1. Go to https://render.com
2. Sign up / Log in (can use GitHub account)
3. Click "New +" → "Web Service"
4. Click "Connect account" to connect GitHub
5. Select your repository
6. Render will auto-detect `render.yaml`
7. Click "Create Web Service"
8. Wait 5-10 minutes for deployment

### Step 5: Test Your API
Once deployed, Render gives you a URL like:
```
https://sign-language-api-xxxx.onrender.com
```

Test it:
```bash
# Health check
curl https://YOUR_APP_NAME.onrender.com/api/health

# Interactive docs
https://YOUR_APP_NAME.onrender.com/api/docs
```

## 🌐 Your API Endpoints

After deployment:
- Health: `https://YOUR_APP_NAME.onrender.com/api/health`
- Model Info: `https://YOUR_APP_NAME.onrender.com/api/model-info`
- Predict: `https://YOUR_APP_NAME.onrender.com/api/predict`
- Docs: `https://YOUR_APP_NAME.onrender.com/api/docs`

## 💰 Pricing

**Free Tier:**
- 750 hours/month free
- Automatic HTTPS
- Spins down after 15 min inactivity
- 512MB RAM

**Paid Tier ($7/month):**
- Always on (no spin down)
- Better performance
- More RAM options

## ⚠️ Important Notes

1. **Cold Start:** On free tier, first request after 15 min takes 30-60 seconds
2. **Model Size:** Your model is 2.61 MB - perfect for Render ✓
3. **Memory:** 512MB is sufficient for your model
4. **Region:** Default is Oregon (can change in render.yaml)

## 🐛 Troubleshooting

### Build Fails
Check Render build logs for errors. Common issues:
- Missing dependencies in requirements_api.txt
- Python version mismatch

### Model Not Loading
Ensure `models/sign_gru_model.h5` is in the repository and pushed to GitHub.

### Out of Memory
Upgrade to paid tier or optimize model.

## 📞 Need Help?

- Render Docs: https://render.com/docs
- Check build logs in Render dashboard
- Verify all files are in GitHub repository

---

**Ready to deploy?** Follow the steps above!
