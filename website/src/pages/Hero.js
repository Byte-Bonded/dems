import React from 'react';
import { Link } from 'react-router-dom';
import { FaArrowRight, FaGithub } from 'react-icons/fa';
import './Hero.css';

function Hero() {
  return (
    <section className="hero">
      <div className="container hero-content">
        <div className="hero-text">
          <h1>Dynamic Energy Management System</h1>
          <p className="hero-subtitle">
            Advanced RL-based energy optimization for intelligent grid management, 
            powered by Reinforcement Learning and real-time monitoring with Prometheus & Grafana
          </p>
          
          <div className="hero-buttons">
            <Link to="/dashboard" className="btn btn-primary">
              View Dashboard <FaArrowRight style={{ marginLeft: '0.5rem' }} />
            </Link>
            <a href="https://github.com/Byte-Bonded/dems" className="btn btn-secondary">
              <FaGithub style={{ marginRight: '0.5rem' }} /> GitHub
            </a>
          </div>

          <div className="hero-stats">
            <div className="stat">
              <div className="stat-number">100+</div>
              <div className="stat-label">Grid Nodes</div>
            </div>
            <div className="stat">
              <div className="stat-number">99.9%</div>
              <div className="stat-label">Uptime</div>
            </div>
            <div className="stat">
              <div className="stat-number">50ms</div>
              <div className="stat-label">Avg Response</div>
            </div>
          </div>
        </div>

        <div className="hero-visual">
          <div className="gradient-box">
            <div className="floating-card">
              <div className="card-header">Energy Status</div>
              <div className="card-value">850 kWh</div>
              <div className="card-progress">
                <div className="progress-bar" style={{ width: '85%' }}></div>
              </div>
            </div>
            <div className="floating-card delayed">
              <div className="card-header">Grid Frequency</div>
              <div className="card-value">50.0 Hz</div>
              <div className="card-status stable">Stable</div>
            </div>
            <div className="floating-card delayed-2">
              <div className="card-header">Optimization</div>
              <div className="card-value">95%</div>
              <div className="card-efficiency">Efficient</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default Hero;
