import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaBars, FaTimes } from 'react-icons/fa';
import './Navbar.css';

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <Link to="/">
            <span className="logo-icon">⚡</span> DEMS
          </Link>
        </div>
        
        <div className={`menu ${isOpen ? 'active' : ''}`}>
          <Link to="/" className="menu-link" onClick={() => setIsOpen(false)}>
            Home
          </Link>
          <a href="#features" className="menu-link" onClick={() => setIsOpen(false)}>
            Features
          </a>
          <Link to="/dashboard" className="menu-link" onClick={() => setIsOpen(false)}>
            Dashboard
          </Link>
          <Link to="/docs" className="menu-link" onClick={() => setIsOpen(false)}>
            Documentation
          </Link>
          <a href="#contact" className="menu-link btn btn-primary" onClick={() => setIsOpen(false)}>
            Get Started
          </a>
        </div>

        <div className="hamburger" onClick={toggleMenu}>
          {isOpen ? <FaTimes /> : <FaBars />}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
