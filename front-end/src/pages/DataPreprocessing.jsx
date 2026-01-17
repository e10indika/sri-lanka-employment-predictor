import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  Alert,
  LinearProgress,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { datasetsAPI } from '../services/api';

export default function DataPreprocessing() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setError(null);
        setSuccess(null);
        setResult(null);
      } else {
        setError('Please select a valid CSV file');
        setSelectedFile(null);
      }
    }
  };

  const handleUpload = async () => {
    try {
      setUploading(true);
      setError(null);
      setSuccess(null);
      
      let response;
      
      if (selectedFile) {
        // Upload with file
        const formData = new FormData();
        formData.append('file', selectedFile);
        response = await datasetsAPI.preprocess(formData);
      } else {
        // Preprocess existing data without file
        response = await datasetsAPI.preprocessExisting();
      }
      
      setSuccess('Data preprocessing completed successfully!');
      setResult(response.data);
      setSelectedFile(null);
      // Reset file input
      const fileInput = document.getElementById('file-input');
      if (fileInput) fileInput.value = '';
    } catch (err) {
      setError('Preprocessing failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Preprocessing
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload a CSV file to preprocess and prepare data for model training.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload New Dataset (Optional)
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <input
            accept=".csv"
            style={{ display: 'none' }}
            id="file-input"
            type="file"
            onChange={handleFileSelect}
          />
          <label htmlFor="file-input">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUploadIcon />}
              fullWidth
              sx={{ mb: 2 }}
            >
              Select CSV File
            </Button>
          </label>
          
          {selectedFile && (
            <Card variant="outlined" sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="body2" color="text.secondary">
                  Selected File:
                </Typography>
                <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                  {selectedFile.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Size: {(selectedFile.size / 1024).toFixed(2)} KB
                </Typography>
              </CardContent>
            </Card>
          )}
        </Box>

        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={uploading}
          fullWidth
          size="large"
        >
          {uploading ? 'Processing...' : selectedFile ? 'Upload and Preprocess' : 'Preprocess Existing Data'}
        </Button>

        {uploading && <LinearProgress sx={{ mt: 2 }} />}
        
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2, textAlign: 'center' }}>
          {selectedFile 
            ? 'Upload and preprocess the selected CSV file' 
            : 'Preprocess the existing dataset on the server'}
        </Typography>
      </Paper>

      {result && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Preprocessing Results
          </Typography>
          
          <List>
            {result.message && (
              <ListItem>
                <ListItemText
                  primary="Status"
                  secondary={result.message}
                />
              </ListItem>
            )}
            <Divider />
            
            {result.total_rows !== undefined && (
              <ListItem>
                <ListItemText
                  primary="Total Rows"
                  secondary={result.total_rows.toLocaleString()}
                />
              </ListItem>
            )}
            
            {result.columns && (
              <>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Columns"
                    secondary={result.columns.join(', ')}
                  />
                </ListItem>
              </>
            )}
            
            {result.preprocessing_steps && (
              <>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Preprocessing Steps Applied"
                    secondary={
                      <Box component="span" sx={{ display: 'block', mt: 1 }}>
                        {result.preprocessing_steps.map((step, index) => (
                          <Typography key={index} variant="body2" component="div">
                            • {step}
                          </Typography>
                        ))}
                      </Box>
                    }
                  />
                </ListItem>
              </>
            )}
            
            {result.file_path && (
              <>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Processed File Path"
                    secondary={result.file_path}
                  />
                </ListItem>
              </>
            )}
          </List>
        </Paper>
      )}
    </Box>
  );
}
