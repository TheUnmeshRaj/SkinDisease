import React, { useState, useContext } from 'react';
import './ImagePage.css';
import axios from 'axios';
import NoteContext from '../NoteContext';

function ImagePage() {
  const { chatHistory, setChatHistory } = useContext(NoteContext);
  const [inputText, setInputText] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleTextSubmit = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/TextAi', { inputText });
      const newMessage = {
        type: 'text',
        query: inputText,
        result: response.data.result,
        treatment: response.data.treatment,
      };
      setChatHistory([...chatHistory, newMessage]);
      setInputText('');
    } catch (error) {
      console.error('Error fetching text result:', error);
    }
    setLoading(false);
  };

  const handleImageSubmit = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await axios.post('http://127.0.0.1:5000/api/ImageAi', formData);
      const newMessage = {
        type: 'image',
        query: image.name,
        result: response.data.result,
        treatment: response.data.treatment,
        imageURL: URL.createObjectURL(image),
      };
      setChatHistory([...chatHistory, newMessage]);
      setImage(null);
    } catch (error) {
      console.error('Error fetching image result:', error);
    }
    setLoading(false);
  };

  const renderChatHistory = () =>
    chatHistory.map((message, index) => (
      <div key={index} className="chat-bubble">
        <div className="query">
          {message.type === 'text' ? (
            <p>User: {message.query}</p>
          ) : (
            <>
              <p>User uploaded:</p>
              <img src={message.imageURL} alt="Uploaded" className="uploaded-img" />
            </>
          )}
        </div>
        <div className="response">
          <p>
            <strong>Result:</strong> {message.result}
          </p>
          <p>
            <strong>Treatment:</strong> {message.treatment}
          </p>
        </div>
      </div>
    ));

  return (
    <div className="chat-container">
      <h1>Skin Disease Detection AI</h1>
      <div className="chat-history">{renderChatHistory()}</div>
      <div className="input-section">
        <input
          type="text"
          placeholder="Enter your symptoms..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="text-input"
        />
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setImage(e.target.files[0])}
          className="file-input"
        />
        <button onClick={handleTextSubmit} disabled={loading} className="send-btn">
        </button>
        <button onClick={handleImageSubmit} disabled={loading || !image} className="send-btn">
          Upload Image
        </button>
      </div>
    </div>
  );
}

export default ImagePage;
