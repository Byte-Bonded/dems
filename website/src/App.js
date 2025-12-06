import React, { useState, useEffect } from 'react';
import './styles/index.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Hero from './pages/Hero';
import Features from './pages/Features';
import Dashboard from './pages/Dashboard';
import Documentation from './pages/Documentation';
import Footer from './components/Footer';

function App() {
  return (
    <Router basename="/dems">
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={
            <>
              <Hero />
              <Features />
            </>
          } />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/docs" element={<Documentation />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
