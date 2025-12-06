import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

function Dashboard() {
  const [energyData, setEnergyData] = useState([]);
  const [gridMetrics, setGridMetrics] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      const mockEnergyData = [
        { time: '00:00', generated: 120, consumed: 100 },
        { time: '04:00', generated: 150, consumed: 130 },
        { time: '08:00', generated: 280, consumed: 200 },
        { time: '12:00', generated: 350, consumed: 320 },
        { time: '16:00', generated: 300, consumed: 280 },
        { time: '20:00', generated: 180, consumed: 150 },
        { time: '23:59', generated: 100, consumed: 90 }
      ];

      const mockMetrics = {
        totalGeneration: 1480,
        totalConsumption: 1270,
        storageLevel: 750,
        gridFrequency: 50.02,
        avgVoltage: 230.5,
        efficiency: 85.8
      };

      setEnergyData(mockEnergyData);
      setGridMetrics(mockMetrics);
      setLoading(false);
    }, 500);
  }, []);

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <div className="container">
        <h1>Energy Management Dashboard</h1>
        
        <div className="metrics-grid grid grid-4">
          <div className="metric-card">
            <div className="metric-label">Total Generated</div>
            <div className="metric-value">{gridMetrics.totalGeneration} kWh</div>
            <div className="metric-change positive">↑ 12.5%</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Total Consumed</div>
            <div className="metric-value">{gridMetrics.totalConsumption} kWh</div>
            <div className="metric-change">Normal</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Storage Level</div>
            <div className="metric-value">{gridMetrics.storageLevel} kWh</div>
            <div className="metric-bar">
              <div className="metric-progress" style={{ width: '75%' }}></div>
            </div>
          </div>
          
          <div className="metric-card">
            <div className="metric-label">Efficiency</div>
            <div className="metric-value">{gridMetrics.efficiency}%</div>
            <div className="metric-change positive">↑ Excellent</div>
          </div>
        </div>

        <div className="charts-grid grid grid-2">
          <div className="chart-container card">
            <h3>Energy Generation vs Consumption</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={energyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="generated" stroke="#E85B7F" strokeWidth={2} />
                <Line type="monotone" dataKey="consumed" stroke="#555555" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-container card">
            <h3>Hourly Grid Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={energyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="generated" fill="#E85B7F" />
                <Bar dataKey="consumed" fill="#D63A60" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="status-section card">
          <h3>System Status</h3>
          <div className="status-grid grid grid-3">
            <div className="status-item">
              <div className="status-label">Grid Frequency</div>
              <div className="status-value">{gridMetrics.gridFrequency} Hz</div>
              <div className="status-indicator healthy">Healthy</div>
            </div>
            
            <div className="status-item">
              <div className="status-label">Voltage</div>
              <div className="status-value">{gridMetrics.avgVoltage} V</div>
              <div className="status-indicator healthy">Stable</div>
            </div>
            
            <div className="status-item">
              <div className="status-label">AI Agent</div>
              <div className="status-value">Active</div>
              <div className="status-indicator healthy">Training</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
