import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Chip,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { modelsAPI, visualizationsAPI } from '../services/api';

export default function CompareModels() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);

  useEffect(() => {
    fetchComparisonData();
  }, []);

  const fetchComparisonData = async () => {
    try {
      setLoading(true);
      const response = await modelsAPI.compare();
      setComparisonData(response.data);
    } catch (err) {
      setError(err.message);
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
    return (
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4">
            Compare Models
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchComparisonData}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        <Alert severity="error">Error loading comparison data: {error}</Alert>
      </Box>
    );
  }

  if (!comparisonData || comparisonData.models.length === 0) {
    return (
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4">
            Compare Models
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchComparisonData}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
        <Alert severity="info">
          No models available for comparison. Train some models first using the "Train Model" page, then click refresh.
        </Alert>
      </Box>
    );
  }

  // Prepare data for charts
  const metricsChartData = comparisonData.models.map(model => ({
    name: model.model_name,
    Accuracy: (model.accuracy * 100).toFixed(2),
    Precision: (model.precision * 100).toFixed(2),
    Recall: (model.recall * 100).toFixed(2),
    'F1 Score': (model.f1_score * 100).toFixed(2),
  }));

  const radarChartData = comparisonData.models.map(model => ({
    metric: model.model_name,
    Accuracy: model.accuracy * 100,
    Precision: model.precision * 100,
    Recall: model.recall * 100,
    'F1 Score': model.f1_score * 100,
  }));

  const trainingTimeData = comparisonData.models.map(model => ({
    name: model.model_name,
    'Training Time (s)': model.training_time.toFixed(2),
  }));

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Compare Models
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Compare the performance of all trained machine learning models.
      </Typography>

      {comparisonData.best_model && (
        <Alert severity="success" sx={{ mb: 3 }}>
          🏆 Best performing model: <strong>{comparisonData.models.find(m => m.model_type === comparisonData.best_model)?.model_name}</strong>
        </Alert>
      )}

      {/* Metrics Comparison Table */}
      <Paper sx={{ mb: 4 }}>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell><strong>Model</strong></TableCell>
                <TableCell align="right"><strong>Accuracy</strong></TableCell>
                <TableCell align="right"><strong>Precision</strong></TableCell>
                <TableCell align="right"><strong>Recall</strong></TableCell>
                <TableCell align="right"><strong>F1 Score</strong></TableCell>
                <TableCell align="right"><strong>CV Mean</strong></TableCell>
                <TableCell align="right"><strong>Training Time</strong></TableCell>
                <TableCell align="center"><strong>Status</strong></TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {comparisonData.models.map((model) => (
                <TableRow 
                  key={model.model_type}
                  sx={{ 
                    bgcolor: model.model_type === comparisonData.best_model ? 'success.light' : 'inherit',
                    '&:hover': { bgcolor: 'action.hover' }
                  }}
                >
                  <TableCell>
                    {model.model_name}
                    {model.model_type === comparisonData.best_model && (
                      <Chip label="Best" color="success" size="small" sx={{ ml: 1 }} />
                    )}
                  </TableCell>
                  <TableCell align="right">{(model.accuracy * 100).toFixed(2)}%</TableCell>
                  <TableCell align="right">{(model.precision * 100).toFixed(2)}%</TableCell>
                  <TableCell align="right">{(model.recall * 100).toFixed(2)}%</TableCell>
                  <TableCell align="right">{(model.f1_score * 100).toFixed(2)}%</TableCell>
                  <TableCell align="right">
                    {model.cv_mean > 0 ? `${(model.cv_mean * 100).toFixed(2)}%` : 'N/A'}
                  </TableCell>
                  <TableCell align="right">{model.training_time.toFixed(2)}s</TableCell>
                  <TableCell align="center">
                    <Chip 
                      label={model.has_visualizations ? 'Complete' : 'Partial'} 
                      color={model.has_visualizations ? 'success' : 'warning'}
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Metrics Bar Chart */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics Comparison
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={metricsChartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="Accuracy" fill="#8884d8" />
            <Bar dataKey="Precision" fill="#82ca9d" />
            <Bar dataKey="Recall" fill="#ffc658" />
            <Bar dataKey="F1 Score" fill="#ff7c7c" />
          </BarChart>
        </ResponsiveContainer>
      </Paper>

      {/* Radar Chart */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Overall Performance Radar
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={[
                {
                  metric: 'Accuracy',
                  ...Object.fromEntries(comparisonData.models.map(m => [m.model_name, m.accuracy * 100]))
                },
                {
                  metric: 'Precision',
                  ...Object.fromEntries(comparisonData.models.map(m => [m.model_name, m.precision * 100]))
                },
                {
                  metric: 'Recall',
                  ...Object.fromEntries(comparisonData.models.map(m => [m.model_name, m.recall * 100]))
                },
                {
                  metric: 'F1 Score',
                  ...Object.fromEntries(comparisonData.models.map(m => [m.model_name, m.f1_score * 100]))
                },
              ]}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" />
                <PolarRadiusAxis domain={[0, 100]} />
                {comparisonData.models.map((model, idx) => (
                  <Radar
                    key={model.model_type}
                    name={model.model_name}
                    dataKey={model.model_name}
                    stroke={['#8884d8', '#82ca9d', '#ffc658'][idx % 3]}
                    fill={['#8884d8', '#82ca9d', '#ffc658'][idx % 3]}
                    fillOpacity={0.3}
                  />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Training Time Chart */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Time Comparison
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={trainingTimeData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="Training Time (s)" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>

      {/* Confusion Matrices Side by Side */}
      <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
        Confusion Matrices
      </Typography>
      <Grid container spacing={3}>
        {comparisonData.models.map((model) => (
          <Grid item xs={12} md={4} key={model.model_type}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {model.model_name}
                  {model.model_type === comparisonData.best_model && (
                    <Chip label="Best" color="success" size="small" sx={{ ml: 1 }} />
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  F1 Score: {(model.f1_score * 100).toFixed(2)}%
                </Typography>
              </CardContent>
              {model.has_visualizations ? (
                <CardMedia
                  component="img"
                  image={visualizationsAPI.getConfusionMatrix(model.model_type)}
                  alt={`${model.model_name} Confusion Matrix`}
                  sx={{ height: 300, objectFit: 'contain', bgcolor: 'grey.100' }}
                />
              ) : (
                <Box 
                  sx={{ 
                    height: 300, 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center',
                    bgcolor: 'grey.100'
                  }}
                >
                  <Typography color="text.secondary">
                    No visualization available
                  </Typography>
                </Box>
              )}
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Cross-Validation Scores */}
      {comparisonData.models.some(m => m.cv_scores && m.cv_scores.length > 0) && (
        <Paper sx={{ p: 3, mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Cross-Validation Scores
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell><strong>Model</strong></TableCell>
                  <TableCell align="right"><strong>Mean</strong></TableCell>
                  <TableCell align="right"><strong>Std Dev</strong></TableCell>
                  <TableCell align="left"><strong>All Scores</strong></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {comparisonData.models
                  .filter(m => m.cv_scores && m.cv_scores.length > 0)
                  .map((model) => (
                    <TableRow key={model.model_type}>
                      <TableCell>{model.model_name}</TableCell>
                      <TableCell align="right">{(model.cv_mean * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">{(model.cv_std * 100).toFixed(2)}%</TableCell>
                      <TableCell align="left">
                        {model.cv_scores.map(s => `${(s * 100).toFixed(1)}%`).join(', ')}
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
}
