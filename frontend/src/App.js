import React, { useState } from "react";
import axios from "axios";
import ComparisonView from './components/ComparisonView';
import "./App.css";

function App() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile1Change = (e) => {
    setFile1(e.target.files[0]);
  };

  const handleFile2Change = (e) => {
    setFile2(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file1 || !file2) return;

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/compare', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/static/sw.js')
        .then(registration => {
          console.log('Service Worker registered with scope:', registration.scope);
        })
        .catch(error => {
          console.error('Service Worker registration failed:', error);
        });
    });
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>File Comparator</h1>
        <form onSubmit={handleSubmit}>
          <div>
            <label>File 1:</label>
            <input type="file" onChange={handleFile1Change} accept=".pdf" />
          </div>
          <div>
            <label>File 2:</label>
            <input type="file" onChange={handleFile2Change} accept=".pdf" />
          </div>
          <button type="submit">Compare</button>
        </form>
        {loading && <p>Processing files...</p>}
        {result && (
          <ComparisonView 
            oldPdf={URL.createObjectURL(file1)}
            newPdf={URL.createObjectURL(file2)}
            changes={result}
          />
        )}
      </header>
    </div>
  );
}

export default App;