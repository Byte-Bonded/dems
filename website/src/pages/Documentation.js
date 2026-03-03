import React from 'react';
import './Documentation.css';

function Documentation() {
  return (
    <div className="documentation">
      <div className="container">
        <h1>Documentation</h1>
        
        <div className="doc-grid grid grid-2">
          <div className="doc-section card">
            <h2>Getting Started</h2>
            <h3>Installation</h3>
            <pre><code>{`# Clone the repository
git clone https://github.com/Byte-Bonded/dems.git
cd dems

# Install dependencies
pip install -r requirements.txt

# Run the API
python -m src.api.main`}</code></pre>

            <h3>Using Docker</h3>
            <pre><code>{`# Build and run with Docker Compose
docker-compose up -d

# Access services
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000`}</code></pre>
          </div>

          <div className="doc-section card">
            <h2>Architecture</h2>
            <h3>Components</h3>
            <ul>
              <li><strong>Core:</strong> Energy and Grid management</li>
              <li><strong>Agent:</strong> RL-based optimization</li>
              <li><strong>API:</strong> FastAPI REST endpoints</li>
              <li><strong>Monitoring:</strong> Prometheus metrics</li>
            </ul>

            <h3>Tech Stack</h3>
            <ul>
              <li>Backend: Python, FastAPI, SQLAlchemy</li>
              <li>RL Framework: Stable-Baselines3, Ray</li>
              <li>Monitoring: Prometheus, Grafana</li>
              <li>Frontend: React, Recharts</li>
              <li>DevOps: Docker, Docker Compose</li>
            </ul>
          </div>

          <div className="doc-section card">
            <h2>API Endpoints</h2>
            <h3>Energy Management</h3>
            <pre><code>{`GET /energy/state
  Get current energy system state

GET /grid/state
  Get current grid state

POST /optimization/predict
  Get optimization prediction from RL agent`}</code></pre>
          </div>

          <div className="doc-section card">
            <h2>Configuration</h2>
            <h3>Environment Variables</h3>
            <pre><code>{`# Copy from .env.example
cp .env.example .env

# Edit with your settings
API_PORT=8000
GRID_SIZE=10
LEARNING_RATE=0.0003
DATABASE_URL=postgresql://...`}</code></pre>
          </div>

          <div className="doc-section card">
            <h2>RL Agent Training</h2>
            <h3>Custom Training Loop</h3>
            <pre><code>{`from src.agent import RLAgent
from src.agent import DEMSEnvironment

env = DEMSEnvironment()
agent = RLAgent(50, 10)

# Training logic here
for episode in range(1000):
    obs = env.reset()
    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)`}</code></pre>
          </div>

          <div className="doc-section card">
            <h2>Monitoring with Prometheus</h2>
            <h3>Metrics Collection</h3>
            <pre><code>{`# Key metrics tracked:
- dems_energy_generated_kwh
- dems_energy_consumed_kwh
- dems_storage_level_kwh
- dems_grid_frequency_hz
- dems_optimization_duration_seconds
- dems_agent_reward`}</code></pre>
          </div>
        </div>

        <div className="doc-resources card">
          <h2>Resources</h2>
          <div className="resources-grid grid grid-3">
            <a href="https://github.com/Byte-Bonded/dems" className="resource-link">
              <div className="resource-icon">📦</div>
              <div className="resource-title">GitHub Repository</div>
              <div className="resource-desc">Source code and issue tracking</div>
            </a>
            <a href="https://stable-baselines3.readthedocs.io" className="resource-link">
              <div className="resource-icon">🤖</div>
              <div className="resource-title">Stable-Baselines3</div>
              <div className="resource-desc">RL algorithms documentation</div>
            </a>
            <a href="https://prometheus.io/docs" className="resource-link">
              <div className="resource-icon">📊</div>
              <div className="resource-title">Prometheus Docs</div>
              <div className="resource-desc">Monitoring and metrics guide</div>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Documentation;
