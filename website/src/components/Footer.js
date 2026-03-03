import React from 'react';
import { FaGithub, FaLinkedin, FaEnvelope } from 'react-icons/fa';
import './Footer.css';

function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-content grid grid-3">
          <div className="footer-section">
            <h3>DEMS</h3>
            <p>Dynamic Energy Management System powered by Reinforcement Learning</p>
          </div>
          
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="/">Home</a></li>
              <li><a href="/docs">Documentation</a></li>
              <li><a href="/dashboard">Dashboard</a></li>
              <li><a href="https://github.com/Byte-Bonded/dems">GitHub</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h4>Connect</h4>
            <div className="social-links">
              <a href="https://github.com/Byte-Bonded/dems" target="_blank" rel="noopener noreferrer">
                <FaGithub />
              </a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                <FaLinkedin />
              </a>
              <a href="mailto:contact@dems.dev">
                <FaEnvelope />
              </a>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p>&copy; {currentYear} Dynamic Energy Management System. All rights reserved.</p>
          <p>Built with ⚡ by Byte-Bonded Team</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
