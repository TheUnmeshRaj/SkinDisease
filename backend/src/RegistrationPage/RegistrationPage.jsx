import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaUser, FaEnvelope, FaLock } from 'react-icons/fa'; // Icons
import { AiFillEye, AiFillEyeInvisible } from 'react-icons/ai'; // Eye toggle
import axios from 'axios';
import './RegistrationPage.css'; // Add new CSS file for custom styles

function RegistrationPage() {
  const [username, setUsername] = useState('');
  const [emailId, setEmailId] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false); // Password visibility toggle
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  // Toggle password visibility
  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  const sendData = (e) => {
    e.preventDefault();

    if (!username || !emailId || !password) {
      setErrorMessage('All fields are required!');
      return;
    }

    setErrorMessage('');
    setSuccessMessage('');
    axios
  .post('http://localhost:5000/Register', { username, emailId, password })
  .then((response) => {
    setSuccessMessage('Registration successful!');
    console.log(response.data.message);
    // Redirect to login or perform other actions
  })
  .catch((error) => {
    setErrorMessage('Registration failed. Please try again.');
    console.error(error);
  });

  };

  return (
    <div className="registration-container">
      <div className="registration-box">
        <h1 className="registration-header">Sign Up</h1>

        {errorMessage && <p className="error-message">{errorMessage}</p>}
        {successMessage && <p className="success-message">{successMessage}</p>}

        {/* Username Input */}
        <div className="input-container">
          <FaUser className="input-icon" />
          <input
            type="text"
            placeholder="Enter your username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="registration-input"
          />
        </div>

        {/* Email Input */}
        <div className="input-container">
          <FaEnvelope className="input-icon" />
          <input
            type="text"
            placeholder="Enter your email"
            value={emailId}
            onChange={(e) => setEmailId(e.target.value)}
            className="registration-input"
          />
        </div>

        {/* Password Input */}
        <div className="input-container">
          <FaLock className="input-icon" />
          <input
            type={showPassword ? 'text' : 'password'}
            placeholder="Enter your password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="registration-input"
          />
          <span className="password-toggle" onClick={togglePasswordVisibility}>
            {showPassword ? <AiFillEyeInvisible /> : <AiFillEye />}
          </span>
        </div>

        {/* Register Button */}
        <button onClick={sendData} className="registration-button">
          Register
        </button>

        {/* Link to Login Page */}
        <div className="registration-footer">
          <p>Already have an account?</p>
          <Link to="/Login" className="login-link">
            Go To Sign In Page
          </Link>
        </div>
      </div>
    </div>
  );
}

export default RegistrationPage;
