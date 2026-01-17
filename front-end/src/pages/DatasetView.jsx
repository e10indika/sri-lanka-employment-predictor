import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import { datasetsAPI } from '../services/api';

export default function DatasetView() {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [sampleData, setSampleData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      const [infoResponse, sampleResponse] = await Promise.all([
        datasetsAPI.getInfo(),
        datasetsAPI.getSample(10),
      ]);
      setDatasetInfo(infoResponse.data);
      setSampleData(sampleResponse.data);
    } catch (err) {
      setError('Failed to load dataset: ' + err.message);
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

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dataset Explorer
      </Typography>

      {datasetInfo && (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Total Rows
                  </Typography>
                  <Typography variant="h4">
                    {datasetInfo.info.rows.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Columns
                  </Typography>
                  <Typography variant="h4">
                    {datasetInfo.info.columns}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            {datasetInfo.info.class_distribution && (
              <>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Employed
                      </Typography>
                      <Typography variant="h4" color="success.main">
                        {datasetInfo.info.class_distribution.Employed.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Typography color="text.secondary" gutterBottom>
                        Unemployed
                      </Typography>
                      <Typography variant="h4" color="error.main">
                        {datasetInfo.info.class_distribution.Unemployed.toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </>
            )}
          </Grid>

          <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
            Sample Data (First 10 Rows)
          </Typography>
          {sampleData && (
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {sampleData.columns.slice(0, 8).map((col) => (
                      <TableCell key={col}>{col}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sampleData.data.map((row, idx) => (
                    <TableRow key={idx}>
                      {sampleData.columns.slice(0, 8).map((col) => (
                        <TableCell key={col}>
                          {row[col] !== null && row[col] !== undefined
                            ? String(row[col])
                            : 'N/A'}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </>
      )}
    </Box>
  );
}
