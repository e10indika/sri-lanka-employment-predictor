# Sri Lanka Employment Predictor - Frontend

Modern React frontend application for the Sri Lanka Employment Predictor system.

## Overview

A responsive web application built with React and Material-UI that provides an intuitive interface for employment prediction, model training, and data visualization.

## Technology Stack

- **React 18.2** - UI library
- **Material-UI 5.14** - Component library
- **React Router 6.20** - Client-side routing
- **Axios 1.6** - HTTP client
- **Recharts 2.10** - Data visualization
- **Vite 5.0** - Build tool and dev server

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable components
│   │   └── Layout.jsx   # Main layout with navigation
│   ├── pages/           # Page components
│   │   ├── Dashboard.jsx       # Model visualizations
│   │   ├── DatasetView.jsx     # Dataset explorer
│   │   ├── TrainModel.jsx      # Training interface
│   │   ├── Predict.jsx         # Prediction form
│   │   └── CompareModels.jsx   # Model comparison
│   ├── services/        # API services
│   │   └── api.js       # API client
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
└── index.html           # HTML template
```

## Installation

### Prerequisites

- Node.js 18 or higher
- npm or yarn package manager

### Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Configure API endpoint:**

Create `.env` file:
```bash
echo "VITE_API_URL=http://localhost:8000" > .env
```

For production:
```bash
echo "VITE_API_URL=https://your-api-domain.com" > .env
```

## Running the Application

### Development Mode

```bash
cd frontend
npm run dev
```

The app will be available at: http://localhost:3000

### Production Build

```bash
cd frontend
npm run build
```

Build output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Features

### 1. Dashboard
- View all trained models
- Display model visualizations (confusion matrix, feature importance, SHAP)
- Switch between different models
- Real-time loading from backend

### 2. Dataset View
- Dataset statistics (total rows, columns, class distribution)
- Sample data preview (first 10 rows)
- Interactive table display
- Column information

### 3. Train Model
- Select model type (XGBoost, Random Forest, etc.)
- Configure cross-validation
- Enable/disable hyperparameter tuning
- Real-time progress tracking
- Display training results
- Automatic model saving

### 4. Predict
- Interactive prediction form
- 7 input features (DISTRICT, SEX, AGE, MARITAL, EDU, Language Profile, Disability)
- Model selection dropdown
- Real-time prediction results
- Probability scores for both classes

### 5. Compare Models
- Side-by-side model comparison
- Performance metrics table
- Interactive bar charts
- Radar chart for multi-dimensional view
- Training time comparison
- Confusion matrices grid
- Cross-validation scores
- Best model highlighting

## API Integration

The frontend communicates with the backend API using Axios. All API calls are centralized in `src/services/api.js`.

### API Configuration

```javascript
// src/services/api.js
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### Available Services

```javascript
import { modelsAPI, trainingAPI, predictionsAPI, datasetsAPI, visualizationsAPI } from './services/api';

// List all models
const models = await modelsAPI.getAll();

// Start training
const job = await trainingAPI.start({
  model_type: 'xgboost',
  perform_cv: true,
  cv_folds: 5
});

// Make prediction
const result = await predictionsAPI.predict({
  model_type: 'xgboost',
  features: { ... }
});
```

## Environment Variables

Create `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

Available variables:
- `VITE_API_URL` - Backend API base URL

## Customization

### Theme

Edit `src/App.jsx` to customize Material-UI theme:

```javascript
import { createTheme, ThemeProvider } from '@mui/material';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});
```

### Navigation

Edit `src/components/Layout.jsx` to add/remove menu items:

```javascript
const menuItems = [
  { text: 'Dashboard', path: '/', icon: <DashboardIcon /> },
  { text: 'My New Page', path: '/my-page', icon: <NewIcon /> },
];
```

### Add New Page

1. Create page component in `src/pages/`:
```javascript
// src/pages/MyPage.jsx
export default function MyPage() {
  return <div>My New Page</div>;
}
```

2. Add route in `src/App.jsx`:
```javascript
import MyPage from './pages/MyPage';

<Route path="/my-page" element={<MyPage />} />
```

3. Add to navigation in `src/components/Layout.jsx`

## Development

### Code Structure

- **Components**: Reusable UI components
- **Pages**: Full page views
- **Services**: API integration and business logic
- **Assets**: Images, fonts, etc.

### Best Practices

1. **Component Organization**: One component per file
2. **API Calls**: Use services layer, not direct axios calls
3. **State Management**: Use React hooks (useState, useEffect)
4. **Error Handling**: Always catch and display errors
5. **Loading States**: Show loading indicators during async operations

### Adding Dependencies

```bash
npm install package-name
```

Common packages:
```bash
# Forms
npm install react-hook-form

# State management
npm install zustand

# Date handling
npm install date-fns
```

## Building for Production

### Build

```bash
npm run build
```

### Optimize Build

Edit `vite.config.js`:
```javascript
export default defineConfig({
  build: {
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
      },
    },
  },
});
```

## Deployment

### GitHub Pages (Recommended for Frontend)

GitHub Pages provides free hosting for static React applications.

#### Step-by-Step Guide

1. **Install gh-pages package:**
```bash
cd frontend
npm install --save-dev gh-pages
```

2. **Update package.json:**

Add these fields:
```json
{
  "homepage": "https://yourusername.github.io/sri-lanka-employment-predictor",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d dist"
  }
}
```

Replace `yourusername` with your GitHub username.

3. **Update vite.config.js:**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/sri-lanka-employment-predictor/', // Add this line
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

4. **Configure Backend URL:**

Update `.env` for production:
```bash
# .env.production
VITE_API_URL=https://your-backend-url.railway.app
```

5. **Deploy:**
```bash
npm run deploy
```

This will:
- Build the production bundle
- Push to `gh-pages` branch
- Deploy to GitHub Pages

6. **Enable GitHub Pages:**
- Go to your repository on GitHub
- Settings → Pages
- Source: Deploy from a branch
- Branch: `gh-pages` / `root`
- Save

Your site will be live at: `https://yourusername.github.io/sri-lanka-employment-predictor`

#### Automated Deployment with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
    paths:
      - 'frontend/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        working-directory: ./frontend
        run: npm install
        
      - name: Build
        working-directory: ./frontend
        run: npm run build
        env:
          VITE_API_URL: ${{ secrets.VITE_API_URL }}
        
      - name: Deploy
        working-directory: ./frontend
        run: |
          git config --global user.name "GitHub Actions"
          git config --global user.email "actions@github.com"
          npm run deploy
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

Add secret in GitHub:
- Settings → Secrets → Actions
- Add `VITE_API_URL` with your backend URL

### Alternative: Netlify

1. **Install Netlify CLI:**
```bash
npm install -g netlify-cli
```

2. **Deploy:**
```bash
cd frontend
npm run build
netlify deploy --prod --dir=dist
```

Or connect GitHub repository:
- Sign up at https://netlify.com
- New site from Git → Choose repository
- Build command: `npm run build`
- Publish directory: `dist`
- Base directory: `frontend`

Add environment variable:
- Site settings → Environment variables
- Key: `VITE_API_URL`
- Value: Your backend URL

### Alternative: Vercel

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Deploy:**
```bash
cd frontend
vercel --prod
```

Or connect GitHub:
- Sign up at https://vercel.com
- New Project → Import from GitHub
- Root Directory: `frontend`
- Framework: Vite
- Build command: `npm run build`
- Output directory: `dist`

Environment variables:
- Settings → Environment Variables
- Add `VITE_API_URL`

### Static Hosting (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /var/www/employment-predictor/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:
```bash
docker build -t employment-predictor-frontend .
docker run -p 3000:80 employment-predictor-frontend
```

### AWS S3 + CloudFront

```bash
# Build
npm run build

# Deploy to S3
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

## Troubleshooting

### API Connection Issues

1. Check backend is running on correct port
2. Verify CORS is configured in backend
3. Check `.env` file has correct API URL
4. Open browser console for errors

### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
```

### Port Conflicts

```bash
# Change port in vite.config.js
export default defineConfig({
  server: {
    port: 3001
  }
});
```

## Performance Optimization

### Code Splitting

```javascript
// Lazy load pages
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));

<Suspense fallback={<CircularProgress />}>
  <Dashboard />
</Suspense>
```

### Image Optimization

- Use WebP format
- Lazy load images
- Use appropriate sizes

### Bundle Analysis

```bash
npm run build
npx vite-bundle-visualizer
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Testing

### Setup Testing (Optional)

```bash
npm install --save-dev vitest @testing-library/react
```

Create test file:
```javascript
// src/pages/Dashboard.test.jsx
import { render, screen } from '@testing-library/react';
import Dashboard from './Dashboard';

test('renders dashboard', () => {
  render(<Dashboard />);
  expect(screen.getByText('Dashboard')).toBeInTheDocument();
});
```

Run tests:
```bash
npm test
```

## Scripts

```json
{
  "dev": "vite",                    // Start dev server
  "build": "vite build",            // Build for production
  "preview": "vite preview",        // Preview production build
  "lint": "eslint src --ext js,jsx" // Lint code
}
```

## License

See main project LICENSE file.

## Support

For issues and questions, see the main project repository.

---

**App Status:** Ready for development
**Version:** 1.0
**Last Updated:** January 15, 2026
