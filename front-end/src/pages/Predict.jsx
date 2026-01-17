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
  ToggleButton,
  ToggleButtonGroup,
  Chip,
  Autocomplete,
} from '@mui/material';
import { modelsAPI, predictionsAPI, datasetsAPI } from '../services/api';

const featureDefinitions = {
  DISTRICT: {
    label: "District",
    type: "select",
    options: {
      "11": "Colombo", "12": "Gampaha", "13": "Kaluthara",
      "21": "Kandy", "22": "Matale", "23": "Nuwara Eliya",
      "31": "Galle", "32": "Mathara", "33": "Hambanthota",
      "41": "Jaffna", "42": "Mannar", "43": "Vavuniya", "44": "Mulathivu", "45": "Kilinochchi",
      "51": "Batticaloa", "52": "Ampara", "53": "Trincomalee",
      "61": "Kurunegala", "62": "Puttalam",
      "71": "Anuradhapura", "72": "Polonnaruwa",
      "81": "Badulla", "82": "Moneragala",
      "91": "Rathnapura", "92": "Kegalle"
    }
  },
  SEX: {
    label: "Sex",
    type: "select",
    options: {
      "1": "Male",
      "2": "Female"
    }
  },
  AGE: {
    label: "Age",
    type: "number",
    min: 15,
    max: 99
  },
  MARITAL: {
    label: "Marital Status",
    type: "select",
    options: {
      "1": "Never Married",
      "2": "Married",
      "3": "Widowed",
      "5": "Divorced/Separated"
    }
  },
  EDU: {
    label: "Education Level",
    type: "select",
    options: {
      "0": "Studying/Studied Grade 1", "1": "Passed Grade 1", "2": "Passed Grade 2",
      "3": "Passed Grade 3", "4": "Passed Grade 4", "5": "Passed Grade 5",
      "6": "Passed Grade 6", "7": "Passed Grade 7", "8": "Passed Grade 8",
      "9": "Passed Grade 9", "10": "Passed Grade 10", "11": "Passed G.C.E (O/L) / N.C.E",
      "12": "Passed Grade 12", "13": "Passed G.C.E (A/L) / H.N.C.E", "14": "Passed G.A.Q. / G.S.Q",
      "15": "Degree", "16": "Postgraduate Degree / Diploma", "17": "Special Educational Institutions",
      "18": "Post Graduate - M", "19": "Post Graduate - PhD"
    }
  },
  Language_Profile_Encoded: {
    label: "Language Profile",
    type: "select",
    options: {
      "0": "ENG", "1": "ENG+SIN", "2": "ENG+SIN+TAMIL",
      "3": "ENG+TAMIL", "4": "None", "5": "SIN",
      "6": "SIN+TAMIL", "7": "TAMIL"
    }
  },
  Disability_Category_Encoded: {
    label: "Disability Category",
    type: "select",
    options: {
      "0": "None", "1": "Mild", "2": "Moderate"
    }
  }
};

export default function Predict() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [mode, setMode] = useState('dataset');
  const [features, setFeatures] = useState({
    DISTRICT: 11,
    SEX: 1,
    AGE: 25,
    MARITAL: 1,
    EDU: 10,
    Language_Profile_Encoded: 3,
    Disability_Category_Encoded: 0,
  });
  const [datasetSamples, setDatasetSamples] = useState([]);
  const [selectedRowIndex, setSelectedRowIndex] = useState('');
  const [totalRows, setTotalRows] = useState(0);
  const [actualValue, setActualValue] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModels();
    loadDatasetSamples();
  }, []);

  const loadModels = async () => {
    try {
      const response = await modelsAPI.getTypes();
      const modelTypes = response.data.model_types || [];
      setModels(modelTypes);
      if (modelTypes.length > 0) {
        setSelectedModel(modelTypes[0].value);
      }
    } catch (err) {
      setError('Failed to load models: ' + err.message);
    }
  };

  const loadDatasetSamples = async () => {
    try {
      const infoResponse = await datasetsAPI.getInfo();
      console.log('Dataset info response:', infoResponse.info);
      const total = infoResponse.info.rows || infoResponse.info.rows || 1000;
      console.log('Setting totalRows to:', total);
      setTotalRows(total);
      // Create array of row numbers for selection dropdown suggestions
      const rowNumbers = Array.from({ length: Math.min(total, 100) }, (_, i) => i);
      setDatasetSamples(rowNumbers);
    } catch (err) {
      console.error('Failed to load dataset info:', err);
      // Try loading first row to get total_rows
      try {
        const rowResponse = await datasetsAPI.getRow(0);
        if (rowResponse.data.total_rows) {
          setTotalRows(rowResponse.data.total_rows);
        }
      } catch (rowErr) {
        console.error('Failed to load row 0:', rowErr);
      }
    }
  };

  const handleModeChange = (event, newMode) => {
    if (newMode !== null) {
      setMode(newMode);
      setPrediction(null);
      setActualValue(null);
      if (newMode === 'manual') {
        setSelectedRowIndex('');
      }
    }
  };

  const handleRowSelect = async (rowNumber) => {
    setSelectedRowIndex(rowNumber);
    if (rowNumber !== '' && rowNumber !== null) {
      try {
        const response = await datasetsAPI.getRow(rowNumber);
        console.log('Full row response:', response.data);
        const rowData = response.data.data;
        console.log('Row data:', rowData);
        console.log('All field names in row data:', Object.keys(rowData));
        // Update totalRows from response if available
        if (response.data.total_rows) {
          setTotalRows(response.data.total_rows);
        }
        const newFeatures = {
          DISTRICT: parseInt(rowData.DISTRICT) || 11,
          SEX: parseInt(rowData.SEX) || 1,
          AGE: parseInt(rowData.AGE) || 25,
          MARITAL: parseInt(rowData.MARITAL) || 1,
          EDU: parseInt(rowData.EDU) || 10,
          Language_Profile_Encoded: parseInt(rowData.Language_Profile_Encoded) || 3,
          Disability_Category_Encoded: parseInt(rowData.Disability_Category_Encoded) || 0,
        };
        setFeatures(newFeatures);
        // Parse EMPLOYMENT_STATUS as integer (0 or 1)
        // The correct field name is Employment_Status_Encoded
        const empStatusValue = rowData.Employment_Status_Encoded;
        console.log('Employment status field value:', empStatusValue);
        const empStatus = !isNaN(parseInt(empStatusValue)) ? parseInt(empStatusValue) : null;
        console.log('Parsed employment status:', empStatus);
        setActualValue(empStatus);
        setPrediction(null);
      } catch (err) {
        console.error('Failed to load row data:', err);
        setError('Failed to load row data: ' + err.message);
      }
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
      // Ensure all feature values are integers and validate
      const intFeatures = {
        DISTRICT: parseInt(features.DISTRICT),
        SEX: parseInt(features.SEX),
        AGE: parseInt(features.AGE),
        MARITAL: parseInt(features.MARITAL),
        EDU: parseInt(features.EDU),
        Language_Profile_Encoded: parseInt(features.Language_Profile_Encoded),
        Disability_Category_Encoded: parseInt(features.Disability_Category_Encoded),
      };
      
      // Validate that all values are valid integers
      for (const [key, value] of Object.entries(intFeatures)) {
        if (isNaN(value)) {
          throw new Error(`Invalid value for ${key}: ${features[key]}`);
        }
      }
      
      console.log('Sending features:', intFeatures);
      
      const response = await predictionsAPI.predict({
        model_type: selectedModel,
        features: intFeatures,
      });
      setPrediction(response.data);
      
      // Log for debugging comparison
      if (actualValue !== null) {
        console.log('Actual vs Predicted:', actualValue, response.data.prediction);
        console.log('Comparison result:', actualValue === response.data.prediction);
      }
    } catch (err) {
      setError('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureChange = (feature, value) => {
    setFeatures((prev) => ({
      ...prev,
      [feature]: parseInt(value) || 0,
    }));
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Make Prediction
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Predict employment status using manual input or actual dataset samples.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'center' }}>
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={handleModeChange}
          aria-label="prediction mode"
        >
          <ToggleButton value="manual" aria-label="manual input">
            Manual Input
          </ToggleButton>
          <ToggleButton value="dataset" aria-label="from dataset">
            From Dataset
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

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
                  <MenuItem key={model.value} value={model.value}>
                    {model.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {mode === 'dataset' && (
              <>
                <Box sx={{ display: 'flex', gap: 1, mb: 2, alignItems: 'flex-start' }}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Row Number"
                    value={selectedRowIndex}
                    onChange={(e) => setSelectedRowIndex(e.target.value)}
                    helperText={totalRows > 0 ? `Enter a row number between 0 and ${totalRows - 1}` : 'Loading...'}
                    inputProps={{ min: 0 }}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        const rowNum = parseInt(selectedRowIndex);
                        if (!isNaN(rowNum) && rowNum >= 0) {
                          if (totalRows > 0 && rowNum >= totalRows) {
                            setError(`Row number must be between 0 and ${totalRows - 1}`);
                          } else {
                            handleRowSelect(rowNum);
                          }
                        }
                      }
                    }}
                  />
                  <Button
                    variant="contained"
                    size="small"
                    onClick={() => {
                      const rowNum = parseInt(selectedRowIndex);
                      if (!isNaN(rowNum) && rowNum >= 0) {
                        if (totalRows > 0 && rowNum >= totalRows) {
                          setError(`Row number must be between 0 and ${totalRows - 1}`);
                        } else {
                          handleRowSelect(rowNum);
                        }
                      } else {
                        setError('Please enter a valid row number');
                      }
                    }}
                    sx={{ 
                      minWidth: '90px',
                      height: '40px',
                      mt: '8px'
                    }}
                  >
                    Load
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    onClick={handlePredict}
                    disabled={loading || !selectedModel}
                    sx={{ 
                      minWidth: '90px',
                      height: '40px',
                      mt: '8px'
                    }}
                  >
                    {loading ? 'Predicting...' : 'Predict'}
                  </Button>
                </Box>

                {actualValue !== null && (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>Actual Employment Status:</strong>{' '}
                      <Chip
                        label={actualValue === 1 ? 'Employed' : 'Unemployed'}
                        color={actualValue === 1 ? 'success' : 'error'}
                        size="small"
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                  </Alert>
                )}
              </>
            )}

            <Divider sx={{ my: 2 }} />

            {Object.keys(features).map((feature) => {
              const def = featureDefinitions[feature];
              const isDisabled = mode === 'dataset' && selectedRowIndex === '';
              
              if (def.type === 'select') {
                return (
                  <FormControl key={feature} fullWidth sx={{ mb: 2 }}>
                    <InputLabel>{def.label}</InputLabel>
                    <Select
                      value={features[feature]}
                      onChange={(e) => handleFeatureChange(feature, e.target.value)}
                      label={def.label}
                      disabled={isDisabled}
                    >
                      {Object.entries(def.options).map(([value, label]) => (
                        <MenuItem key={value} value={parseInt(value)}>
                          {label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                );
              }
              
              return (
                <TextField
                  key={feature}
                  fullWidth
                  label={def.label}
                  type="number"
                  value={features[feature]}
                  onChange={(e) => handleFeatureChange(feature, e.target.value)}
                  sx={{ mb: 2 }}
                  disabled={isDisabled}
                  inputProps={{
                    min: def.min || 0,
                    max: def.max || 999,
                  }}
                />
              );
            })}

            <Button
              variant="contained"
              fullWidth
              onClick={handlePredict}
              disabled={loading || !selectedModel || (mode === 'dataset' && selectedRowIndex === '')}
              sx={{ mt: 2 }}
            >
              {loading ? 'Predicting...' : 'Predict Employment Status'}
            </Button>
            
            {mode === 'dataset' && selectedRowIndex === '' && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'center' }}>
                Please select a data row first
              </Typography>
            )}
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

                {mode === 'dataset' && actualValue !== null && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Box sx={{ my: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Actual Status:
                          </Typography>
                          <Chip
                            label={actualValue === 1 ? 'Employed' : 'Unemployed'}
                            color={actualValue === 1 ? 'success' : 'error'}
                            sx={{ mt: 0.5 }}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">
                            Predicted Status:
                          </Typography>
                          <Chip
                            label={prediction.prediction_label}
                            color={prediction.prediction === 1 ? 'success' : 'error'}
                            sx={{ mt: 0.5 }}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <Divider sx={{ my: 1 }} />
                          <Typography variant="body2" align="center">
                            <strong>
                              {actualValue === prediction.prediction ? (
                                <span style={{ color: 'green' }}>✓ Correct Prediction</span>
                              ) : (
                                <span style={{ color: 'red' }}>✗ Incorrect Prediction</span>
                              )}
                            </strong>
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>
                  </>
                )}

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

      {/* SHAP Explanation Section */}
      {prediction && prediction.shap_force_plot && (
        <Box sx={{ mt: 4 }}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                SHAP Force Plot - Feature Contribution
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                This visualization shows how each feature contributed to the prediction. 
                Features pushing towards "Employed" are shown in red, while features pushing towards "Unemployed" are shown in blue.
              </Typography>
              <Box
                component="img"
                src={`data:image/png;base64,${prediction.shap_force_plot}`}
                alt="SHAP Force Plot"
                sx={{
                  width: '100%',
                  height: 'auto',
                  mt: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                }}
              />
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
}
