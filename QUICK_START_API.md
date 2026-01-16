# Backend API Quick Reference

## 🚀 Start the Server

```bash
./start_server.sh
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Local Network: http://YOUR_IP:8000

## 📍 Key Endpoints

### Train a Model
```bash
curl -X POST "http://localhost:8000/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xgboost", "perform_cv": true, "cv_folds": 5}'
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/api/predictions/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "xgboost",
    "features": {
      "DISTRICT": 1,
      "SEX": 1,
      "AGE": 35,
      "MARITAL": 2,
      "EDU": 3,
      "Language Profile": 1,
      "Disability": 0
    }
  }'
```

### List Models
```bash
curl http://localhost:8000/api/models/
```

### Compare Models
```bash
curl http://localhost:8000/api/models/compare
```

### Dataset Info
```bash
curl http://localhost:8000/api/datasets/info
```

## 🎯 Available Models

- `xgboost` (Recommended)
- `random_forest`
- `gradient_boosting`
- `logistic_regression`
- `svm`
- `neural_network`

## 🌐 External Access

### Find Your IP
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### Access from Network
```
http://YOUR_IP:8000
```

### Internet Access (ngrok)
```bash
ngrok http 8000
```

## 🔧 Troubleshooting

### Port in Use
```bash
lsof -i :8000
kill -9 $(lsof -t -i:8000)
```

### Reinstall Dependencies
```bash
cd backend
```

---

**Frontend Location:** `/Users/indika/External/GitHub/sri-lanka-employment-predictor-UI`

**Full Docs:** See [README.md](README.md)
