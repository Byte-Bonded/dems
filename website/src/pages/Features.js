import React from 'react';
import { FaRobot, FzBarChart, FaShieldAlt, FaZap, FaDatabase, FaCloud } from 'react-icons/fa';
import './Features.css';

function Features() {
  const features = [
    {
      icon: <FaRobot />,
      title: 'RL-Based Optimization',
      description: 'Advanced Reinforcement Learning agents continuously optimize energy distribution and storage management in real-time.'
    },
    {
      icon: <FzBarChart />,
      title: 'Real-Time Monitoring',
      description: 'Prometheus metrics and Grafana dashboards provide comprehensive visibility into your energy system performance.'
    },
    {
      icon: <FaShieldAlt />,
      title: 'Grid Stability',
      description: 'Intelligent algorithms maintain grid frequency and voltage stability, preventing outages and ensuring reliability.'
    },
    {
      icon: <FaZap />,
      title: 'Smart Distribution',
      description: 'Dynamic load balancing across grid nodes minimizes losses and maximizes efficiency.'
    },
    {
      icon: <FaDatabase />,
      title: 'Data Persistence',
      description: 'PostgreSQL and Redis integration for reliable data storage and fast caching of critical metrics.'
    },
    {
      icon: <FaCloud />,
      title: 'Scalable Architecture',
      description: 'Docker-containerized microservices enable easy scaling and deployment across multiple environments.'
    }
  ];

  return (
    <section id="features" className="features">
      <div className="container">
        <div className="section-header">
          <h2>Powerful Features</h2>
          <p>Everything you need for intelligent energy management</p>
        </div>

        <div className="features-grid grid grid-3">
          {features.map((feature, index) => (
            <div key={index} className="feature-card card">
              <div className="feature-icon">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Features;
