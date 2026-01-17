import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DatasetView from './pages/DatasetView';
import DataPreprocessing from './pages/DataPreprocessing';
import TrainModel from './pages/TrainModel';
import Predict from './pages/Predict';
import CompareModels from './pages/CompareModels';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const basename = import.meta.env.VITE_BASE_PATH || '/sri-lanka-employment-predictor-UI';
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router basename={basename}>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/dataset" element={<DatasetView />} />
            <Route path="/preprocess" element={<DataPreprocessing />} />
            <Route path="/train" element={<TrainModel />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/compare" element={<CompareModels />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
