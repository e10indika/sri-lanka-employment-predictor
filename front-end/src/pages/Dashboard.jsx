import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Button,
  IconButton,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { modelsAPI, visualizationsAPI } from '../services/api';

export default function Dashboard() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const response = await modelsAPI.getAll();
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].model_type);
      }
      setError(null);
    } catch (err) {
      console.error('Failed to load models:', err);
      setError('Failed to load models. Make sure the backend API is running at ' + (import.meta.env.VITE_API_URL || 'https://ffb6a6813594.ngrok-free.app'));
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (models.length === 0) {
    return (
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4">
            Model Dashboard
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadModels}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        <Alert severity="info">
          No trained models found. Please train a model first, then click refresh.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h4">
          Model Dashboard
        </Typography>
        <IconButton onClick={loadModels} disabled={loading} color="primary">
          <RefreshIcon />
        </IconButton>
      </Box>
      <Typography variant="body1" color="text.secondary" paragraph>
        View trained models' performance metrics and explainability visualizations.
      </Typography>

      <Box sx={{ mb: 3 }}>
        <FormControl fullWidth>
          <InputLabel>Select Model</InputLabel>
          <Select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            label="Select Model"
          >
            {models.map((model) => (
              <MenuItem key={model.model_type} value={model.model_type}>
                {model.model_name} ({model.filename})
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {selectedModel && (
        <Grid container spacing={3} key={selectedModel}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Confusion Matrix
                </Typography>
                <Box
                  component="img"
                  key={`cm-${selectedModel}`}
                  src={visualizationsAPI.getConfusionMatrix(selectedModel)}
                  alt="Confusion Matrix"
                  sx={{ width: '100%', height: 'auto' }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'block';
                  }}
                />
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ display: 'none', mt: 2 }}
                >
                  Visualization not available. Train and evaluate this model first.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Feature Importance
                </Typography>
                <Box
                  component="img"
                  key={`fi-${selectedModel}`}
                  src={visualizationsAPI.getFeatureImportance(selectedModel)}
                  alt="Feature Importance"
                  sx={{ width: '100%', height: 'auto' }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'block';
                  }}
                />
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ display: 'none', mt: 2 }}
                >
                  Visualization not available.
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  SHAP Summary
                </Typography>
                <Box
                  component="img"
                  key={`shap-${selectedModel}`}
                  src={visualizationsAPI.getShapSummary(selectedModel)}
                  alt="SHAP Summary"
                  sx={{ width: '100%', height: 'auto', maxHeight: '600px', objectFit: 'contain' }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'block';
                  }}
                />
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ display: 'none', mt: 2 }}
                >
                  Visualization not available.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      <Box sx={{ mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Available Models
        </Typography>
        <Grid container spacing={2}>
          {models.map((model) => (
            <Grid item xs={12} md={4} key={model.model_type}>
              <Card>
                <CardContent>
                  <Typography variant="h6">{model.model_name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Type: {model.model_type}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    File: {model.filename}
                  </Typography>
                  <Typography variant="body2" color={model.has_visualizations ? 'success.main' : 'warning.main'}>
                    {model.has_visualizations ? '✅ Visualizations available' : '⚠️ Needs evaluation'}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
}
