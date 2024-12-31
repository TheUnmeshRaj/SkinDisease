import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaUser, FaLock } from 'react-icons/fa'; // Icons for username and password
import { AiFillEye, AiFillEyeInvisible } from 'react-icons/ai'; // Icons for show/hide password
import axios from 'axios';
import './LoginPage.css';

function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false); // For password visibility
  const [errorMessage, setErrorMessage] = useState('');

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const handleLogin = (e) => {
    e.preventDefault();

    if (!username || !password) {
      setErrorMessage('Both fields are required!');
      return;
    }

    setErrorMessage('');

    axios
  .post('http://localhost:5000/Login', { username, password })
  .then((response) => {
    console.log('Login successful:', response.data.message);
    // Handle successful login (maybe store token, redirect, etc.)
  })
  .catch((err) => {
    console.error('Login failed:', err);
    setErrorMessage('Invalid username or password!');
  });

  };

  return (
    <div className="login-container">
      
      <div className="login-box">
        <h1 className="login-header">Welcome Back</h1>

        {errorMessage && <p className="error-message">{errorMessage}</p>}

        {/* Username Field */}
        <div className="input-container">
          <FaUser className="input-icon" />
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="login-input"
          />
        </div>

        {/* Password Field */}
        <div className="input-container">
          <FaLock className="input-icon" />
          <input
            type={showPassword ? 'text' : 'password'}
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="login-input"
          />
          <span className="password-toggle" onClick={togglePasswordVisibility}>
            {showPassword ? <AiFillEyeInvisible /> : <AiFillEye />}
          </span>
        </div>

        {/* Login Button */}
        <button onClick={handleLogin} className="login-button">
          Login
        </button>

        {/* Link to Registration Page */}
        <div className="login-footer">
          <p>Don't have an account?</p>
          <Link to="/Register" className="register-link">
            Register Here
          </Link>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
