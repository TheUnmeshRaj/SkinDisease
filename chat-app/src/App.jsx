import React, { useContext, useEffect, useState } from 'react';
// import { Link } from 'react-router-dom';
import './App.css';
import './components/HeaderFile/HeaderStyle.css';

import HeaderFile from './components/HeaderFile/HeaderFile';
import SocialMedia from './components/SocialMedia/SocialMedia';
import InputText from './components/InputText/InputText';
import NoteContext from './components/NoteContext';

function App() {
  const [mode, setMode] = useState('Dark'); // Default mode is Dark
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const { userId, setUserId, mainRes, setMainRes } = useContext(NoteContext);

  // Toggle dropdown menu visibility
  const toggleDropdown = () => setIsDropdownOpen((prev) => !prev);

  // Save userId and mainRes to localStorage on change
  useEffect(() => {
    localStorage.setItem('id', JSON.stringify(userId));
  }, [userId]);

  useEffect(() => {
    localStorage.setItem('res', JSON.stringify(mainRes));
  }, [mainRes]);

  // Apply background color based on the current mode
  useEffect(() => {
    document.body.style.backgroundColor = mode === 'Dark' ? '#343541' : 'white';
  }, [mode]);

  // Toggle theme mode
  const changeMode = () => setMode((prevMode) => (prevMode === 'Light' ? 'Dark' : 'Light'));

  // Reusable button component
  const renderButton = () => (
    <button
      className={`app-login-${userId !== '.' ? 'LoggedIn' : 'Login'}`}
      onClick={userId !== '.' ? toggleDropdown : null}
    >
      {userId !== '.' ? 'U' : 'Login'}
    </button>
  );

  return (
    <div className={`App-${mode}`}>
      <div className={`social-media-${mode}`}>
        <div className="right-items">
          <SocialMedia Theme={mode} />
        </div>
        <div className="center-items">
          <HeaderFile mode={mode} />
        </div>
        <div className="left-items">
          <button className={`mode-${mode}`} onClick={changeMode}></button>
        </div>
      </div>
      <InputText mode={mode} />
    </div>
  );
}

export default App;
