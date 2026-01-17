import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Paper,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  Grid,
  Chip,
  TextField,
} from '@mui/material';
import { modelsAPI, trainingAPI, visualizationsAPI } from '../services/api';

export default function TrainModel() {
  const [availableModels, setAvailableModels] = useState([]);
  const [modelType, setModelType] = useState('');
  const [selectedModels, setSelectedModels] = useState([]);
  const [performCV, setPerformCV] = useState(true);
  const [cvFolds, setCvFolds] = useState(5);
  const [performTuning, setPerformTuning] = useState(false);
  const [useParamGrid, setUseParamGrid] = useState(false);
  const [paramGrid, setParamGrid] = useState({
    max_depth: '5,10',
    learning_rate: '0.05,0.1',
    n_estimators: '200,300',
  });
  const [training, setTraining] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModelTypes();
  }, []);

  useEffect(() => {
    let interval;
    if (jobId && training) {
      interval = setInterval(checkStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, training]);

  const loadModelTypes = async () => {
    try {
      const response = await modelsAPI.getTypes();
      const modelTypes = response.data.model_types || [];
      setAvailableModels(modelTypes);
      if (modelTypes.length > 0) {
        setModelType(modelTypes[0].value);
        setSelectedModels([modelTypes[0].value]);
      }
    } catch (err) {
      setError('Failed to load model types: ' + err.message);
    }
  };

  const handleSelectAll = () => {
    if (selectedModels.length === availableModels.length) {
      setSelectedModels([]);
    } else {
      setSelectedModels(availableModels.map(m => m.value));
    }
  };

  const handleModelToggle = (modelValue) => {
    setSelectedModels(prev => {
      if (prev.includes(modelValue)) {
        return prev.filter(m => m !== modelValue);
      } else {
        return [...prev, modelValue];
      }
    });
  };

  const checkStatus = async () => {
    try {
      const response = await trainingAPI.getStatus(jobId);
      setStatus(response.data);

      if (response.data.status === 'completed' || response.data.status === 'failed') {
        setTraining(false);
        if (response.data.status === 'failed') {
          setError(response.data.error);
        }
      }
    } catch (err) {
      console.error('Failed to check status:', err);
    }
  };

  const waitForCompletion = async (jobId) => {
    return new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        try {
          const response = await trainingAPI.getStatus(jobId);
          setStatus(response.data);

          if (response.data.status === 'completed') {
            clearInterval(interval);
            resolve();
          } else if (response.data.status === 'failed') {
            clearInterval(interval);
            setError(response.data.error);
            reject(new Error(response.data.error));
          }
        } catch (err) {
          clearInterval(interval);
          reject(err);
        }
      }, 2000);
    });
  };

  const handleTrain = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model to train');
      return;
    }

    try {
      setTraining(true);
      setError(null);
      setStatus(null);

      // Train all selected models sequentially
      for (const model of selectedModels) {
        const payload = {
          model_type: model,
          perform_tuning: performTuning,
          use_param_grid: useParamGrid,
        };
        
        // Add CV options if enabled
        if (performCV) {
          payload.cv_folds = cvFolds;
        }
        
        // Add param_grid if use_param_grid is enabled
        if (useParamGrid && performTuning) {
          const parsedGrid = {};
          Object.entries(paramGrid).forEach(([key, value]) => {
            // Parse comma-separated values to array of numbers
            const values = value.split(',').map(v => {
              const trimmed = v.trim();
              return isNaN(trimmed) ? trimmed : parseFloat(trimmed);
            });
            parsedGrid[key] = values;
          });
          payload.param_grid = parsedGrid;
        }

        const response = await trainingAPI.start(payload);

        setJobId(response.data.job_id);
        setModelType(model); // Update current model being trained
        
        // Wait for this model to complete before starting the next one
        await waitForCompletion(response.data.job_id);
      }
      
      setTraining(false);
    } catch (err) {
      setError('Failed to complete training: ' + err.message);
      setTraining(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Train Employment Prediction Model
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Train new models to predict employment status based on Sri Lankan labour force statistics.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Configuration
            </Typography>

            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                <Typography variant="subtitle1">Select Models to Train</Typography>
                <Button
                  size="small"
                  onClick={handleSelectAll}
                  disabled={training}
                >
                  {selectedModels.length === availableModels.length ? 'Deselect All' : 'Select All'}
                </Button>
              </Box>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {availableModels.map((model) => (
                  <FormControlLabel
                    key={model.value}
                    control={
                      <Checkbox
                        checked={selectedModels.includes(model.value)}
                        onChange={() => handleModelToggle(model.value)}
                        disabled={training}
                      />
                    }
                    label={model.label}
                  />
                ))}
              </Box>
            </Box>

            <FormControlLabel
              control={
                <Checkbox
                  checked={performCV}
                  onChange={(e) => setPerformCV(e.target.checked)}
                  disabled={training}
                />
              }
              label="Perform Cross-Validation"
            />

            {performCV && (
              <FormControl fullWidth sx={{ mb: 2, mt: 1 }}>
                <InputLabel>CV Folds</InputLabel>
                <Select
                  value={cvFolds}
                  onChange={(e) => setCvFolds(e.target.value)}
                  label="CV Folds"
                  disabled={training}
                >
                  {[3, 4, 5, 7, 10].map((n) => (
                    <MenuItem key={n} value={n}>
                      {n} Folds
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <FormControlLabel
              control={
                <Checkbox
                  checked={performTuning}
                  onChange={(e) => setPerformTuning(e.target.checked)}
                  disabled={training}
                />
              }
              label="Hyperparameter Tuning (slower)"
            />
            
            {performTuning && (
              <Box sx={{ ml: 4, mt: 1 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={useParamGrid}
                      onChange={(e) => setUseParamGrid(e.target.checked)}
                      disabled={training}
                    />
                  }
                  label="Use Custom Parameter Grid"
                />
                
                {useParamGrid && (
                  <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary" gutterBottom>
                      Enter comma-separated values for each parameter:
                    </Typography>
                    {Object.entries(paramGrid).map(([param, value]) => (
                      <TextField
                        key={param}
                        fullWidth
                        size="small"
                        label={param}
                        value={value}
                        onChange={(e) => setParamGrid(prev => ({ ...prev, [param]: e.target.value }))}
                        disabled={training}
                        placeholder="e.g., 5,10,15"
                        sx={{ mt: 1 }}
                        helperText={`Values to test for ${param}`}
                      />
                    ))}
                  </Box>
                )}
              </Box>
            )}

            <Button
              variant="contained"
              fullWidth
              onClick={handleTrain}
              disabled={training || selectedModels.length === 0}
              sx={{ mt: 3 }}
            >
              {training ? 'Training...' : `Train ${selectedModels.length} Model${selectedModels.length !== 1 ? 's' : ''}`}
            </Button>
          </Paper>
        </Grid>

        {/* Training Progress Section */}
        {status && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Progress
                </Typography>

                <Box sx={{ my: 2 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      mb: 1,
                    }}
                  >
                    <Typography variant="body2">{status.message}</Typography>
                    <Typography variant="body2">{status.progress}%</Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={status.progress}
                  />
                </Box>

                <Chip
                  label={status.status.toUpperCase()}
                  color={
                    status.status === 'completed'
                      ? 'success'
                      : status.status === 'failed'
                      ? 'error'
                      : 'primary'
                  }
                  sx={{ mt: 2 }}
                />
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Training Details Section */}
        {status && status.results && status.results.training_details && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Training Details
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Typography variant="body2" color="text.secondary">
                      Model: <strong>{status.results.model_name}</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Train Samples: <strong>{status.results.training_details.train_samples}</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Test Samples: <strong>{status.results.training_details.test_samples}</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Features: <strong>{status.results.training_details.features}</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      CV Folds: <strong>{status.results.training_details.cv_folds || 'N/A'}</strong>
                    </Typography>
                  </Grid>
                  {status.results.training_details.hyperparameter_tuning && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="text.secondary">
                        Hyperparameter Tuning: <strong>Enabled</strong>
                      </Typography>
                      {status.results.training_details.tuned_parameters && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                          Tuned: {status.results.training_details.tuned_parameters.join(', ')}
                        </Typography>
                      )}
                    </Grid>
                  )}
                  {status.results.training_time && (
                    <Grid item xs={12}>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        Training Time: <strong>{status.results.training_time.toFixed(2)}s</strong>
                      </Typography>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Performance Metrics Section */}
        {status && status.results && status.results.metrics && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Accuracy: <strong>{(status.results.metrics.accuracy * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      F1 Score (Weighted): <strong>{(status.results.metrics.f1_weighted * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Precision (Weighted): <strong>{(status.results.metrics.precision_weighted * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Recall (Weighted): <strong>{(status.results.metrics.recall_weighted * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Precision (Macro): <strong>{(status.results.metrics.precision_macro * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      Recall (Macro): <strong>{(status.results.metrics.recall_macro * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      F1 Score (Macro): <strong>{(status.results.metrics.f1_macro * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Cross-Validation Scores Section */}
        {status && status.results && status.results.cv_scores && status.results.cv_scores.length > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Cross-Validation Scores
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      CV Mean: <strong>{(status.results.cv_mean * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2">
                      CV Std: <strong>{(status.results.cv_std * 100).toFixed(2)}%</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="caption" color="text.secondary">
                      Fold Scores: {status.results.cv_scores.map(s => (s * 100).toFixed(2) + '%').join(', ')}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Best Hyperparameters Section */}
        {status && status.results && status.results.best_hyperparameters && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Best Hyperparameters
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(status.results.best_hyperparameters).map(([key, value]) => (
                    <Grid item xs={6} key={key}>
                      <Typography variant="body2" color="text.secondary">
                        {key}: <strong>{typeof value === 'number' ? value : String(value)}</strong>
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Feature Importance Section */}
        {status && status.results && status.results.feature_importance && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Feature Importance
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(status.results.feature_importance)
                    .sort((a, b) => b[1] - a[1])
                    .map(([feature, importance]) => (
                      <Grid item xs={6} md={3} key={feature}>
                        <Typography variant="body2">
                          {feature}: <strong>{importance}</strong>
                        </Typography>
                      </Grid>
                    ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        )}


      </Grid>

      {/* Visualizations Section */}
      {status && status.status === 'completed' && status.results && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h5" gutterBottom>
            Model Visualizations
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Confusion Matrix
                  </Typography>
                  <Box
                    component="img"
                    key={`cm-${modelType}`}
                    src={visualizationsAPI.getConfusionMatrix(modelType)}
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
                    sx={{ display: 'none', mt: 2, textAlign: 'center' }}
                  >
                    Visualization not yet available
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
                    key={`fi-${modelType}`}
                    src={visualizationsAPI.getFeatureImportance(modelType)}
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
                    sx={{ display: 'none', mt: 2, textAlign: 'center' }}
                  >
                    Visualization not yet available
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    SHAP Summary
                  </Typography>
                  <Box
                    component="img"
                    key={`shap-${modelType}`}
                    src={visualizationsAPI.getShapSummary(modelType)}
                    alt="SHAP Summary"
                    sx={{ width: '100%', height: 'auto' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ display: 'none', mt: 2, textAlign: 'center' }}
                  >
                    Visualization not yet available
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    LIME Explanation
                  </Typography>
                  <Box
                    component="img"
                    key={`lime-${modelType}`}
                    src={visualizationsAPI.getLimeExplanation(modelType)}
                    alt="LIME Explanation"
                    sx={{ width: '100%', height: 'auto' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ display: 'none', mt: 2, textAlign: 'center' }}
                  >
                    Visualization not yet available
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Partial Dependence Plot
                  </Typography>
                  <Box
                    component="img"
                    key={`pdp-${modelType}`}
                    src={visualizationsAPI.getPartialDependence(modelType)}
                    alt="Partial Dependence"
                    sx={{ width: '100%', height: 'auto', maxHeight: '600px', objectFit: 'contain' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ display: 'none', mt: 2, textAlign: 'center' }}
                  >
                    Visualization not yet available
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>
      )}
    </Box>
  );
}
