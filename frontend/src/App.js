import React from "react";
import "./App.css";
import { FiVideo, FiImage } from "react-icons/fi";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Age and Gender Detection</h1>
        <div className="options">
          <button className="btn" onClick={() => window.location.reload()}>
            <FiVideo size={20} />
            Start Webcam Detection
          </button>
          <a href="/upload">
            <button className="btn">
              <FiImage size={20} />
              Upload an Image
            </button>
          </a>
        </div>
        <div className="video-feed">
          <img src="http://127.0.0.1:5000/video_feed" alt="Video Feed" />
        </div>
      </header>
    </div>
  );
}

export default App;
