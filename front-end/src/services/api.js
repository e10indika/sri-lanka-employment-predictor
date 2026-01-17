import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://ffb6a6813594.ngrok-free.app';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Models API
export const modelsAPI = {
  getAll: () => api.get('/api/models/'),
  getTypes: () => api.get('/api/models/types'),
  getConfigs: () => api.get('/api/models/configs'),
  getDetails: (modelType) => api.get(`/api/models/${modelType}/details`),
  compare: () => api.get('/api/models/compare'),
  delete: (modelType) => api.delete(`/api/models/${modelType}`),
};

// Training API
export const trainingAPI = {
  start: (data) => api.post('/api/training/start', data),
  getStatus: (jobId) => api.get(`/api/training/status/${jobId}`),
  listJobs: () => api.get('/api/training/jobs'),
  deleteJob: (jobId) => api.delete(`/api/training/jobs/${jobId}`),
};

// Predictions API
export const predictionsAPI = {
  predict: (data) => api.post('/api/predictions/predict', data),
  batchPredict: (modelType, data) => 
    api.post(`/api/predictions/batch-predict?model_type=${modelType}`, data),
  getFeatures: () => api.get('/api/predictions/features'),
};

// Datasets API
export const datasetsAPI = {
  getInfo: () => api.get('/api/datasets/info'),
  getSample: (n = 10) => api.get(`/api/datasets/sample?n=${n}`),
  getRow: (rowNumber) => api.get(`/api/datasets/row/${rowNumber}`),
  getColumnInfo: (columnName) => api.get(`/api/datasets/column/${columnName}`),
  getCorrelation: () => api.get('/api/datasets/correlation'),
  preprocess: (formData) => {
    return axios.post(`${API_BASE_URL}/api/datasets/preprocess`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  preprocessExisting: () => api.post('/api/datasets/preprocess'),
};

// Visualizations API
export const visualizationsAPI = {
  getConfusionMatrix: (modelType) => 
    `${API_BASE_URL}/api/visualizations/${modelType}/confusion-matrix`,
  getFeatureImportance: (modelType) => 
    `${API_BASE_URL}/api/visualizations/${modelType}/feature-importance`,
  getShapSummary: (modelType) => 
    `${API_BASE_URL}/api/visualizations/${modelType}/shap-summary`,
  getLimeExplanation: (modelType) => 
    `${API_BASE_URL}/api/visualizations/${modelType}/lime-explanation`,
  getPartialDependence: (modelType) => 
    `${API_BASE_URL}/api/visualizations/${modelType}/partial-dependence`,
  getStatus: (modelType) => api.get(`/api/visualizations/${modelType}/status`),
};

export default api;
