import React, { useState, useContext } from 'react';
import './InputText.css';
import axios from 'axios';
import NoteContext from '../NoteContext';
import googlePoint from './assets/googlePoint.png';

import magnifyingGlass from '../images/searchIcon.png';
import camera from '../images/camera.png';

const headingTxt = "Welcome to our Skin Disease Detection AI";
const InitialText = "Your reliable companion for skin health! Our advanced AI tool uses cutting-edge technology to accurately identify and classify a wide range of skin conditions. Whether you're concerned about a new spot, rash, or other issues, our system provides quick and reliable insights, helping you stay informed and proactive about your skin health.";

function InputText(props) {
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('Enter some Query');
  const [heading, setHeading] = useState(headingTxt);
  const [loading, setLoading] = useState(false);
  const [explainPara, setExplainPara] = useState(InitialText);
  const { userId } = useContext(NoteContext);
  const { mainRes, setMainRes } = useContext(NoteContext);
  const currUserId = userId;
const [selectedImage, setSelectedImage] = useState(null);


  const fetchRes = () => {
    const data = { inputText };
    axios.post("http://127.0.0.1:5001/api/TextAi", data)
      .then((response) => {
        const result = response.data;
        console.log("API Response:", response.data); // Check response structure
        const dataPack = { 
          query: inputText, 
          res: result.disease  || "Disease might be", // Fallback if no result
          desc: result.treatment || "No treatment found", // Fallback if no treatment
          doctors: result.doctors || []  // Add doctors' info to the result
        };
        setMainRes((prevMainRes) => [...prevMainRes, dataPack]);
        setOutputText(''); 
      })
      .catch((error) => {
        setOutputText('Error fetching result');
        console.error("Error:", error);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  const storeRes = () => {
    const data = { mainRes, currUserId };
    axios.post("http://localhost:3002", data)
      .then((res) => {
        console.log(res.data.message);
      })
      .catch((err) => {
        console.log("Error storing data:", err);
      });
  };

  const handleClick = () => {
    if (inputText === '') {
      setHeading('Please Enter Your Symptoms');
      setExplainPara('');
    } else {
      setLoading(true);
      setOutputText('Fetching Result'); 
      fetchRes(); 
      storeRes();
      setInputText('');
    }
  };
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      console.log("Selected image:", file);
    }
  };
  const changeTxt = (e) => {
    setInputText(e.target.value);
  };

  // Function to format the treatment plan
  const formatTreatment = (treatmentText) => {
    const paragraphs = treatmentText.split("\n").filter(paragraph => paragraph.trim() !== "");

    return paragraphs.map((para, index) => {
      if (para.includes(":")) {
        return (
          <div key={index}>
            <h3>{para.split(":")[0]}</h3>
            <p>{para.split(":")[1]}</p>
          </div>
        );
      } else {
        return <p key={index}>{para}</p>;
      }
    });
  };

  const renderFunc = () => {
    if (mainRes.length === 0) {
      return (
        <>
          <h1 className="hed-ini">{heading}</h1>
          <p className="explain-ini">{explainPara}</p>
        </>
      );
    }
    return mainRes.map((item, index) => (
      <React.Fragment key={index}>
        <div className="query-box">
          <p className="query">{item.query}</p>
        </div>
        <div className="res-desc-box">
          <h1 className="res">{`RESULT: ${item.res.toUpperCase()}`}</h1>
          {/* Format the treatment here */}
          <div className="treatment">{formatTreatment(item.desc)}</div>
        </div>
        {item.doctors.length > 0 && (
          <div className="doctor-details">
            <h2>Recommended Doctors</h2>
            {item.doctors.map((doctor, docIndex) => (
              <div key={docIndex} className="doctor-card">
                <h3>{doctor.name}</h3>
                <p><strong>Qualifications:</strong> {doctor.qualifications || "Not available"}</p>
                {/* <p><strong>Specializations:</strong> {doctor.specializations || "Not available"}</p> */}
                {/* <p><strong>Experience:</strong> {doctor.experience || "Not available"}</p> */}
                <p>
  {doctor.clinics && doctor.clinics.length > 0 ? (
    <div>
      <h3>Clinics</h3>
      {doctor.clinics.map((clinic, index) => {
        const baseAddress =
          "R.+V.+College+of+Engineering,+Mysore+Rd,+RV+Vidyaniketan,+Post,+Bengaluru,+Karnataka+560059,+India";
        const encodedClinic = encodeURIComponent(clinic);
        const googleMapsURL = `https://www.google.com/maps/dir/${baseAddress}/${encodedClinic}/@12.9160575,77.5054058,15z/data=!4m14!4m13!1m5!1m1!1s0x3bae3ee159ba3729:0x75a3463d510cf26e!2m2!1d77.4987012!2d12.9237228!1m5!1m1!1s0x3bae14824624a8b9:0x6d9e06c14ac07d57!2m2!1d77.6515047!2d12.9174417!3e3?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D`;

        return (
          <div key={index} style={{ marginBottom: "8px" }}>
            <a
              href={googleMapsURL}
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src={googlePoint}
                alt="Map Icon"
                style={{ width: "20px", position: "relative", top: "-2px", left:"3px"}}
                className="map-icon"
              />
            </a>
            {clinic}
          </div>
        );
      })}
    </div>
  ) : (
    "Not available"
  )}
</p>

                <a href={doctor.link} target="_blank" rel="noopener noreferrer" className="doctor-link">View Profile</a>
              </div>
            ))}
          </div>
        )}
      </React.Fragment>
    ));
  };

  return (
    <div className={`body-container-${props.mode}`}>
      <div className="res-box">
        {renderFunc()}
      </div>
      <div className="ip-search">
  {/* Upload Button for Image */}
  
  <button
  className="upload-button"  onClick={() => window.open('http://localhost:3001')}>
  <img src={camera} alt="Upload" className="upload-icon" />
 </button>

 

  {/* Hidden Image Input */}
  <input
    type="file"
    id="imageInput"
    style={{ display: 'none' }}
    accept="image/*"
    onChange={(e) => handleFileUpload(e)}
  />
  
  {/* Text Input */}
  <input
    type="text"
    value={inputText}
    onChange={changeTxt}
    placeholder="Enter your query"
    className="ip"
  />
  
  {/* Search Button */}
  <button onClick={handleClick} disabled={loading} className="search-button">
    {loading ? (
      <div className="loading-spinner">
        <div className="spinner"></div>
      </div>
    ) : (
      <img src={magnifyingGlass} alt="Search" className="magnifying-glass" />
    )}
  </button>
</div>

    </div>
  );
}

export default InputText;
