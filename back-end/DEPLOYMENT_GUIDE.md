# Deployment Guide - GitHub Pages + Backend Service

Complete guide for deploying the Sri Lanka Employment Predictor to production.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   Frontend (GitHub Pages)               │
│   https://yourusername.github.io        │
│   - Static React App                    │
│   - Free hosting                        │
└──────────────┬──────────────────────────┘
               │ HTTPS/REST API
               ▼
┌─────────────────────────────────────────┐
│   Backend API (Railway/Render/Heroku)  │
│   https://your-app.railway.app          │
│   - FastAPI Server                      │
│   - ML Models                           │
└─────────────────────────────────────────┘
```

## Quick Deployment (TL;DR)

### 1. Deploy Backend (Railway - 5 minutes)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
cd backend
railway login
railway init
railway up

# Note the URL: https://your-app.railway.app
```

### 2. Deploy Frontend (GitHub Pages - 3 minutes)

```bash
cd frontend

# Install gh-pages
npm install --save-dev gh-pages

# Add to package.json:
# "homepage": "https://yourusername.github.io/sri-lanka-employment-predictor"
# "scripts": { "deploy": "gh-pages -d dist" }

# Update .env.production
echo "VITE_API_URL=https://your-app.railway.app" > .env.production

# Deploy
npm run deploy
```

Visit: `https://yourusername.github.io/sri-lanka-employment-predictor`

---

## Detailed Step-by-Step Guide

## Part 1: Backend Deployment

### Option A: Railway (Recommended - Free $5/month credit)

**Why Railway?**
- ✅ Free tier available
- ✅ Easy deployment
- ✅ Automatic HTTPS
- ✅ Good for Python/FastAPI
- ✅ Built-in PostgreSQL if needed

**Steps:**

1. **Create Railway Account:**
   - Visit https://railway.app
   - Sign up with GitHub

2. **Install Railway CLI:**
```bash
npm install -g @railway/cli
```

3. **Deploy Backend:**
```bash
cd backend
railway login
railway init

# Follow prompts:
# - Create new project
# - Name: employment-predictor-api

railway up
```

4. **Configure Environment:**
```bash
# Add environment variables (optional)
railway variables set API_HOST=0.0.0.0
railway variables set API_PORT=8000
```

5. **Get Deployment URL:**
```bash
railway open
```

Copy your URL (e.g., `https://employment-predictor-api-production.up.railway.app`)

6. **Enable CORS for your GitHub Pages domain:**

Update `backend/api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yourusername.github.io"  # Add this
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Redeploy:
```bash
railway up
```

### Option B: Render (Free Tier)

**Steps:**

1. **Create Account:** https://render.com

2. **Create New Web Service:**
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Select your repository

3. **Configure:**
   - Name: `employment-predictor-api`
   - Region: Choose closest to your users
   - Branch: `main`
   - Root Directory: `backend`
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables:**
   - Add `PYTHON_VERSION` = `3.10`

5. **Create Service** (free tier)

Your API will be at: `https://employment-predictor-api.onrender.com`

**Note:** Free tier sleeps after inactivity. First request may be slow.

### Option C: Heroku (Free tier discontinued, $5/month minimum)

```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login
heroku login

# Create app
cd backend
heroku create employment-predictor-api

# Create Procfile
echo "web: uvicorn api.main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open
heroku open
```

---

## Part 2: Frontend Deployment to GitHub Pages

### Prerequisites

- GitHub account
- Repository pushed to GitHub
- Backend deployed and URL obtained

### Step-by-Step

1. **Prepare package.json:**

Edit `frontend/package.json`:

```json
{
  "name": "employment-predictor-frontend",
  "version": "1.0.0",
  "homepage": "https://YOURUSERNAME.github.io/REPONAME",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  },
  "dependencies": {
    // ... existing dependencies
  },
  "devDependencies": {
    // ... existing devDependencies
    "gh-pages": "^6.1.0"
  }
}
```

Replace:
- `YOURUSERNAME` with your GitHub username
- `REPONAME` with your repository name

2. **Install gh-pages:**

```bash
cd frontend
npm install --save-dev gh-pages
```

3. **Update vite.config.js:**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/REPONAME/',  // Add this - must match repo name
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

4. **Create Production Environment File:**

```bash
cd frontend

# Create .env.production
cat > .env.production << EOF
VITE_API_URL=https://your-actual-backend-url.railway.app
EOF
```

Replace with your actual backend URL from Step 1.

5. **Update API Configuration:**

Edit `frontend/src/services/api.js` to ensure it uses env variable:

```javascript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

6. **Deploy:**

```bash
cd frontend
npm run deploy
```

This will:
- Build the production bundle
- Create/update `gh-pages` branch
- Push to GitHub

7. **Enable GitHub Pages:**

- Go to your repository on GitHub
- Click **Settings**
- Click **Pages** (left sidebar)
- Under "Source":
  - Branch: `gh-pages`
  - Folder: `/ (root)`
- Click **Save**

8. **Wait for Deployment:**

GitHub will build and deploy (1-2 minutes).

Your site will be live at:
```
https://YOURUSERNAME.github.io/REPONAME
```

### Verify Deployment

1. Visit your GitHub Pages URL
2. Check browser console for errors
3. Test API connection (should connect to your backend)
4. Try making a prediction

---

## Part 3: Automated Deployment with GitHub Actions

### Setup CI/CD Pipeline

1. **Create workflow file:**

```bash
mkdir -p .github/workflows
```

Create `.github/workflows/deploy-frontend.yml`:

```yaml
name: Deploy Frontend to GitHub Pages

on:
  push:
    branches: [ main ]
    paths:
      - 'frontend/**'
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    defaults:
      run:
        working-directory: ./frontend
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: './frontend/package-lock.json'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Build
        run: npm run build
        env:
          VITE_API_URL: ${{ secrets.VITE_API_URL }}
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./frontend/dist
          cname: your-custom-domain.com  # Optional: if using custom domain
```

2. **Add Backend URL Secret:**

- Go to GitHub repository
- Settings → Secrets and variables → Actions
- Click "New repository secret"
- Name: `VITE_API_URL`
- Value: Your backend URL (e.g., `https://your-app.railway.app`)
- Click "Add secret"

3. **Push to trigger deployment:**

```bash
git add .
git commit -m "Add GitHub Actions workflow"
git push origin main
```

4. **Monitor deployment:**

- Go to Actions tab in your repository
- Watch the workflow run
- Once complete, check your site

---

## Part 4: Custom Domain (Optional)

### Add Custom Domain to GitHub Pages

1. **Purchase domain** (e.g., from Namecheap, Google Domains)

2. **Configure DNS:**

Add these records:

```
Type    Name    Value
A       @       185.199.108.153
A       @       185.199.109.153
A       @       185.199.110.153
A       @       185.199.111.153
CNAME   www     yourusername.github.io
```

3. **Configure GitHub Pages:**

- Repository → Settings → Pages
- Custom domain: `your-domain.com`
- Click Save
- Wait for DNS check (may take 24-48 hours)
- Enable "Enforce HTTPS"

4. **Update vite.config.js:**

```javascript
base: '/',  // Change from '/reponame/' to '/'
```

5. **Redeploy:**

```bash
cd frontend
npm run deploy
```

### Add Custom Domain to Backend

**Railway:**
```bash
railway domain
```

**Render:**
- Go to dashboard → Your service
- Settings → Custom Domain
- Add domain and configure DNS

---

## Troubleshooting

### Frontend Issues

**404 on refresh:**

Create `frontend/public/_redirects` (for Netlify):
```
/*  /index.html  200
```

Or for GitHub Pages, the router handles this automatically with `HashRouter`.

**CORS errors:**

Update backend CORS to include your GitHub Pages domain:
```python
allow_origins=["https://yourusername.github.io"]
```

**API not connecting:**

Check `.env.production` has correct backend URL:
```bash
cat frontend/.env.production
# Should show: VITE_API_URL=https://your-backend.com
```

**Blank page:**

Check `vite.config.js` base path matches repository name:
```javascript
base: '/repository-name/',
```

### Backend Issues

**Module not found:**

Ensure you're running from project root:
```bash
cd backend
python -m uvicorn api.main:app
```

**Railway deployment fails:**

Check `requirements.txt` is in backend directory:
```bash
ls backend/requirements.txt
```

**API returns 500:**

Check Railway logs:
```bash
railway logs
```

---

## Cost Breakdown

### Free Tier Options

| Service | Frontend | Backend | Limits |
|---------|----------|---------|--------|
| **GitHub Pages** | ✅ Free | ❌ N/A | 1 GB storage, 100 GB bandwidth/month |
| **Railway** | ❌ N/A | ✅ $5 credit/month | 500 hours, sleeps after inactivity |
| **Render** | ✅ Free | ✅ Free | Sleeps after 15min inactivity |
| **Netlify** | ✅ Free | ❌ N/A | 100 GB bandwidth |
| **Vercel** | ✅ Free | ❌ N/A | 100 GB bandwidth |

### Recommended Setup (Free)

- **Frontend:** GitHub Pages (free, no limits for public repos)
- **Backend:** Railway ($5 credit/month, enough for hobby projects)

Total: **Free** (within Railway $5 credit)

---

## Monitoring

### Frontend

Add analytics to `frontend/index.html`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Backend

Add logging in Railway:
```bash
railway logs --follow
```

---

## Security Checklist

- [ ] Backend CORS configured for production domain only
- [ ] API rate limiting implemented (optional)
- [ ] Sensitive data not in frontend code
- [ ] Environment variables used for API URL
- [ ] HTTPS enabled on both frontend and backend
- [ ] Error messages don't expose sensitive information

---

## Maintenance

### Update Frontend

```bash
cd frontend
# Make changes
git add .
git commit -m "Update frontend"
git push origin main  # Triggers GitHub Actions deployment
```

Or manually:
```bash
npm run deploy
```

### Update Backend

```bash
cd backend
# Make changes
railway up  # For Railway
```

Or push to trigger automatic deployment:
```bash
git add .
git commit -m "Update backend"
git push origin main
```

---

## Complete Example

**Your deployed app:**
- Frontend: `https://yourusername.github.io/sri-lanka-employment-predictor`
- Backend: `https://employment-predictor-api.up.railway.app`
- API Docs: `https://employment-predictor-api.up.railway.app/docs`

Users visit the frontend, which makes API calls to the backend transparently!

---

**Deployment Status:** Production Ready  
**Total Cost:** Free (within limits)  
**Deployment Time:** ~15 minutes  
**Last Updated:** January 15, 2026
