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
} from '@mui/material';
import { modelsAPI, trainingAPI } from '../services/api';

const modelConfigs = {
  xgboost: 'XGBoost',
  random_forest: 'Random Forest',
  decision_tree: 'Decision Tree',
  gradient_boosting: 'Gradient Boosting',
  naive_bayes: 'Naive Bayes',
  logistic_regression: 'Logistic Regression',
};

export default function TrainModel() {
  const [modelType, setModelType] = useState('xgboost');
  const [performCV, setPerformCV] = useState(true);
  const [cvFolds, setCvFolds] = useState(5);
  const [performTuning, setPerformTuning] = useState(false);
  const [training, setTraining] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let interval;
    if (jobId && training) {
      interval = setInterval(checkStatus, 2000);
    }
    return () => clearInterval(interval);
  }, [jobId, training]);

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

  const handleTrain = async () => {
    try {
      setTraining(true);
      setError(null);
      setStatus(null);

      const response = await trainingAPI.start({
        model_type: modelType,
        perform_cv: performCV,
        cv_folds: cvFolds,
        perform_tuning: performTuning,
      });

      setJobId(response.data.job_id);
    } catch (err) {
      setError('Failed to start training: ' + err.message);
      setTraining(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Train Employment Prediction Model
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Train a new model to predict employment status based on Sri Lankan labour force statistics.
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

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Model Type</InputLabel>
              <Select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                label="Model Type"
                disabled={training}
              >
                {Object.entries(modelConfigs).map(([key, label]) => (
                  <MenuItem key={key} value={key}>
                    {label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

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

            <Button
              variant="contained"
              fullWidth
              onClick={handleTrain}
              disabled={training}
              sx={{ mt: 3 }}
            >
              {training ? 'Training...' : 'Start Training'}
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          {status && (
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

                {status.results && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Results
                    </Typography>
                    <Typography variant="body2">
                      Model: {status.results.model_name}
                    </Typography>
                    <Typography variant="body2">
                      Accuracy: {(status.results.metrics.accuracy * 100).toFixed(2)}%
                    </Typography>
                    <Typography variant="body2">
                      F1-Score: {(status.results.metrics.f1_weighted * 100).toFixed(2)}%
                    </Typography>
                    {status.results.cv_scores && (
                      <Typography variant="body2">
                        CV Accuracy:{' '}
                        {(
                          status.results.cv_scores.reduce((a, b) => a + b, 0) /
                          status.results.cv_scores.length
                        ).toFixed(4)}
                      </Typography>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}
