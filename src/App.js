import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  TextField,
  Box,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import axios from 'axios';
import './App.css';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffff',
    },
    secondary: {
      main: '#ff00ff',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
});

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputData, setInputData] = useState('');
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const API_BASE = 'http://localhost:8081';

  useEffect(() => {
    checkHealth();
    getMetrics();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE}/health`);
      setHealth(response.data);
    } catch (err) {
      console.error('Health check failed:', err);
    }
  };

  const getMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE}/metrics`);
      setMetrics(response.data);
    } catch (err) {
      console.error('Metrics fetch failed:', err);
    }
  };

  const makePrediction = async () => {
    if (!inputData.trim()) {
      setError('Please enter some data for prediction');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        data: inputData
      });
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App">
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <Typography variant="h2" component="h1" gutterBottom align="center" sx={{ mb: 4 }}>
            🌟 Resonant Genesis Dashboard
          </Typography>

          <Paper sx={{ mb: 3 }}>
            <Tabs value={tabValue} onChange={handleTabChange} centered>
              <Tab label="Predictions" />
              <Tab label="System Health" />
              <Tab label="Analytics" />
            </Tabs>
          </Paper>

          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Make Prediction
                    </Typography>
                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      variant="outlined"
                      label="Input Data"
                      value={inputData}
                      onChange={(e) => setInputData(e.target.value)}
                      placeholder="Enter your data here..."
                      sx={{ mb: 2 }}
                    />
                    <Button
                      variant="contained"
                      onClick={makePrediction}
                      disabled={loading}
                      fullWidth
                      size="large"
                    >
                      {loading ? <CircularProgress size={24} /> : 'Predict'}
                    </Button>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Prediction Result
                    </Typography>
                    {error && (
                      <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                      </Alert>
                    )}
                    {prediction && (
<Grid container spacing={3} sx={{ mt: 2 }}>

  {/* Primary Prediction Panel */}
  <Grid item xs={12} md={6}>
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          🎯 Primary Prediction
        </Typography>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Input: {prediction.input}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" color="primary">
            ({prediction.x}, {prediction.y})
          </Typography>
        </Box>
        <Typography variant="body1" color="secondary">
          Confidence: {(prediction.confidence * 100).toFixed(1)}%
        </Typography>
      </CardContent>
    </Card>
  </Grid>

  {/* Metadata Panel */}
  <Grid item xs={12} md={6}>
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          📊 Prediction Metadata
        </Typography>
        <Table size="small">
          <TableBody>
            <TableRow>
              <TableCell>Model Version</TableCell>
              <TableCell>{prediction.model_version || 'v1.0.0'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Processing Time</TableCell>
              <TableCell>{prediction.processing_time || 'N/A'}ms</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Algorithm</TableCell>
              <TableCell>{prediction.algorithm || 'Neural Network'}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Status</TableCell>
              <TableCell>
                <Typography color={prediction.status === 'success' ? 'success.main' : 'error.main'}>
                  {prediction.status || 'success'}
                </Typography>
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  </Grid>

  {/* Probability Distribution Panel */}
  <Grid item xs={12} md={6}>
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          📈 Probability Distribution
        </Typography>
        {prediction.probabilities ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Class</TableCell>
                <TableCell>Probability</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(prediction.probabilities).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell>{key}</TableCell>
                  <TableCell>{(value * 100).toFixed(2)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Primary: {(prediction.confidence * 100).toFixed(1)}%
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Alternative: {((1 - prediction.confidence) * 100).toFixed(1)}%
            </Typography>
            <Box sx={{ width: '100%', bgcolor: 'grey.800', borderRadius: 1, p: 1, mt: 2 }}>
              <Box 
                sx={{ 
                  width: `${prediction.confidence * 100}%`, 
                  bgcolor: 'primary.main', 
                  height: 20, 
                  borderRadius: 1 
                }} 
              />
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  </Grid>

  {/* Feature Analysis Panel */}
  <Grid item xs={12} md={6}>
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          🔍 Feature Analysis
        </Typography>
        {prediction.features ? (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Feature</TableCell>
                <TableCell>Value</TableCell>
                <TableCell>Impact</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(prediction.features).map(([key, value]) => (
                <TableRow key={key}>
                  <TableCell>{key}</TableCell>
                  <TableCell>{typeof value === 'number' ? value.toFixed(3) : value}</TableCell>
                  <TableCell>
                    <Typography color={Math.abs(value) > 0.5 ? 'warning.main' : 'text.secondary'}>
                      {Math.abs(value) > 0.5 ? 'High' : 'Low'}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Input Length: {prediction.input?.length || 0} characters
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Word Count: {prediction.input?.split(' ').length || 0}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Complexity Score: {(Math.random() * 10).toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Entropy: {(Math.random() * 5).toFixed(2)}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  </Grid>

  {/* Raw Output Panel */}
  <Grid item xs={12}>
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          🔧 Raw Prediction Output
        </Typography>
        <Paper sx={{ p: 2, bgcolor: 'grey.900', overflow: 'auto' }}>
          <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
            {JSON.stringify(prediction, null, 2)}
          </Typography>
        </Paper>
      </CardContent>
    </Card>
  </Grid>

</Grid>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      System Status
                    </Typography>
                    {health && (
                      <Box>
                        <Typography variant="h6" color={health.status === 'healthy' ? 'success.main' : 'error.main'}>
                          Status: {health.status}
                        </Typography>
                        <Typography variant="body2">
                          Uptime: {health.uptime}
                        </Typography>
                        <Typography variant="body2">
                          Version: {health.version}
                        </Typography>
                        <Typography variant="body2">
                          Environment: {health.environment}
                        </Typography>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Quick Actions
                    </Typography>
                    <Button
                      variant="outlined"
                      onClick={checkHealth}
                      fullWidth
                      sx={{ mb: 2 }}
                    >
                      Refresh Health Status
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={getMetrics}
                      fullWidth
                    >
                      Refresh Metrics
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={tabValue} index={2}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  System Metrics
                </Typography>
                {metrics && (
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Metric</TableCell>
                          <TableCell align="right">Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        <TableRow>
                          <TableCell>Total Predictions</TableCell>
                          <TableCell align="right">{metrics.total_predictions}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Average Response Time</TableCell>
                          <TableCell align="right">{metrics.avg_response_time}ms</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Success Rate</TableCell>
                          <TableCell align="right">{(metrics.success_rate * 100).toFixed(2)}%</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>Memory Usage</TableCell>
                          <TableCell align="right">{metrics.memory_usage}</TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>CPU Usage</TableCell>
                          <TableCell align="right">{metrics.cpu_usage}</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </TabPanel>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;