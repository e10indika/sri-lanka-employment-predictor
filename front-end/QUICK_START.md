# Quick Start Guide

## Separate Frontend Project Setup Complete! ✓

The frontend has been successfully separated into its own project at:
`/Users/indika/External/GitHub/sri-lanka-employment-predictor-UI`

---

## Running the Application

### Option 1: Using the Start Script (Recommended)

```bash
cd /Users/indika/External/GitHub/sri-lanka-employment-predictor-UI
./START.sh
```

### Option 2: Using npm directly

```bash
cd /Users/indika/External/GitHub/sri-lanka-employment-predictor-UI
npm run dev
```

The frontend will be available at: **http://localhost:3000**

---

## Prerequisites

Before running the frontend, ensure:

1. **Node.js 18+** is installed
2. **Dependencies are installed** (already done):
   ```bash
   npm install
   ```
3. **Backend server is running** at http://localhost:8000

---

## Starting the Backend

The frontend needs the backend API to be running. Start it from the main project:

```bash
 /Users/indika/External/GitHub/sri-lanka-employment-predictor/backend
./start_server.sh
```

Or:
## Separate Fronts/indika/External/GitHub/sri-lanka-employment-predictor/backend
uvicorn api.main:app --reload --port 8000
```

---

## Configuration

### API URL

The frontend is configured to connect to the backend via the `.env` file:

```env
VITE_API_URL=http://localhost:8000
```

To change the ./START.sh
```

### Option 2: Using npm directly

```bash
cd /Usefiguration

The fr
```bash
cd /Users/indi0 by defaulcd /Uschange it, edit `vite.config.js`:

```javascript
export default defineConfig({
  
Trver: {
    port: 3001  // Change to your preferred port
  }
});
```

---

## Features

1. **Dashboard**: View model visualizations (confusion matrix, feature   ```bash
   npm install
   ```
3. **Backend sedataset with statistics and samples
3. **Train Model**: Train new ML models with configurable parameters
4. **Predict**: Make predictions using trained models
5. **Compare Models**: Side-by-side model performance comparison

---

## Troubleshooting

### Frontend won't start

1. Ensure you're in the correct directory:
   ```bash
   cd /Users/indika/External/GitHub/sri-lanka-employment-predic```

---

## Configuration

### API URL
:
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

### API Connection Errors

1. Verify backend is running:
   ```bash
   curl http://localhost:8000/docs
   ```

2. Check `.env` file contains:
   ```
   VITE_API_URL=http://localhost:8000
   ```

```javascript
export default defineConfig({
  
Trver: {
    poWorkflow

1. **Start Backend** (Terminal 1):
   ```bash
   cd /Users/indika/External/GitHub/sri-lanka-employment-predictor/backend
   ./start_server.sh
   ```

2. **Start Frontend** (Terminal 2):
   ```bash
   cd /Users/indika/External/GitHub/sri-lanka-employment-predictor-UI
   npm run dev
   ```

3. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

---

**Last Updated:** January 15, 2026
