import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Paper,
  Grid,
  Alert,
  Card,
  CardContent,
  Divider,
} from '@mui/material';
import { modelsAPI, predictionsAPI } from '../services/api';

const featureLabels = {
  DISTRICT: 'District (1-25)',
  SEX: 'Sex (1=Male, 2=Female)',
  AGE: 'Age',
  MARITAL: 'Marital Status (1-6)',
  EDU: 'Education Level (1-20)',
  Language_Profile_Encoded: 'Language Proficiency (0-7)',
  Disability_Category_Encoded: 'Disability Category (0-6)',
};

export default function Predict() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [features, setFeatures] = useState({
    DISTRICT: 1,
    SEX: 1,
    AGE: 25,
    MARITAL: 1,
    EDU: 10,
    Language_Profile_Encoded: 3,
    Disability_Category_Encoded: 0,
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const response = await modelsAPI.getAll();
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0].model_type);
      }
    } catch (err) {
      setError('Failed to load models: ' + err.message);
    }
  };

  const handlePredict = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await predictionsAPI.predict({
        model_type: selectedModel,
        features: features,
      });
      setPrediction(response.data);
    } catch (err) {
      setError('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureChange = (feature, value) => {
    setFeatures((prev) => ({
      ...prev,
      [feature]: parseFloat(value) || 0,
    }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Make Prediction
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Enter individual characteristics to predict employment status.
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
              Input Features
            </Typography>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Model</InputLabel>
              <Select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="Model"
              >
                {models.map((model) => (
                  <MenuItem key={model.model_type} value={model.model_type}>
                    {model.model_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Divider sx={{ my: 2 }} />

            {Object.keys(features).map((feature) => (
              <TextField
                key={feature}
                fullWidth
                label={featureLabels[feature] || feature}
                type="number"
                value={features[feature]}
                onChange={(e) => handleFeatureChange(feature, e.target.value)}
                sx={{ mb: 2 }}
                inputProps={{
                  step: feature === 'AGE' ? 1 : 1,
                  min: 0,
                }}
              />
            ))}

            <Button
              variant="contained"
              fullWidth
              onClick={handlePredict}
              disabled={loading || !selectedModel}
              sx={{ mt: 2 }}
            >
              {loading ? 'Predicting...' : 'Predict Employment Status'}
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          {prediction && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Prediction Result
                </Typography>

                <Box sx={{ my: 3, textAlign: 'center' }}>
                  <Typography
                    variant="h3"
                    color={
                      prediction.prediction === 1 ? 'success.main' : 'error.main'
                    }
                  >
                    {prediction.prediction_label}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Using {prediction.model_name}
                  </Typography>
                </Box>

                <Divider sx={{ my: 2 }} />

                <Typography variant="h6" gutterBottom>
                  Probabilities
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      mb: 1,
                    }}
                  >
                    <Typography>Employed:</Typography>
                    <Typography fontWeight="bold">
                      {(prediction.probabilities.Employed * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography>Unemployed:</Typography>
                    <Typography fontWeight="bold">
                      {(prediction.probabilities.Unemployed * 100).toFixed(2)}%
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}
