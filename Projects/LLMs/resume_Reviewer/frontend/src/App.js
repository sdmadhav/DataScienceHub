import React, { useState } from 'react';
import { Upload, FileText, Briefcase, CheckCircle, AlertCircle, TrendingUp, Award, Target, Lightbulb, Download } from 'lucide-react';
import './App.css';

export default function ResumeAnalyzer() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescFile, setJobDescFile] = useState(null);
  const [jobDescText, setJobDescText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');

  const handleResumeUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      setResumeFile(file);
    } else {
      alert('Please upload a PDF file');
    }
  };

  const handleJobDescUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'text/plain') {
      setJobDescFile(file);
      const reader = new FileReader();
      reader.onload = (event) => {
        setJobDescText(event.target.result);
      };
      reader.readAsText(file);
    } else {
      alert('Please upload a TXT file');
    }
  };

  const analyzeResume = async () => {
    if (!resumeFile || (!jobDescFile && !jobDescText)) {
      alert('Please upload both resume and job description');
      return;
    }

    setLoading(true);
    setActiveTab('results');

    const formData = new FormData();
    formData.append('resume', resumeFile);
    if (jobDescFile) {
      formData.append('job_description', jobDescFile);
    } else {
      formData.append('job_description_text', jobDescText);
    }

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      alert('Error analyzing resume: ' + error.message);
      setActiveTab('upload');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return '#10b981';
    if (score >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const getScoreLabel = (score) => {
    if (score >= 0.8) return 'Excellent Match';
    if (score >= 0.6) return 'Good Match';
    if (score >= 0.4) return 'Fair Match';
    return 'Needs Improvement';
  };

  const downloadReport = () => {
    if (!results) return;
    
    const reportContent = `
RESUME ANALYSIS REPORT
=====================

Match Score: ${(results.match_score * 100).toFixed(1)}%
Assessment: ${getScoreLabel(results.match_score)}

${results.review}

---
Generated on: ${new Date().toLocaleString()}
    `.trim();

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'resume-analysis-report.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-container">
              <Award size={32} color="white" />
            </div>
            <div>
              <h1 className="title">AI Resume Analyzer</h1>
              <p className="subtitle">Powered by AI • Match Score • Career Insights</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Tabs */}
        <div className="tabs">
          <button
            onClick={() => setActiveTab('upload')}
            className={`tab ${activeTab === 'upload' ? 'tab-active' : ''}`}
          >
            <Upload size={16} style={{ marginRight: '8px' }} />
            Upload Files
          </button>
          <button
            onClick={() => setActiveTab('results')}
            disabled={!results}
            className={`tab ${activeTab === 'results' ? 'tab-active' : ''} ${!results ? 'tab-disabled' : ''}`}
          >
            <TrendingUp size={16} style={{ marginRight: '8px' }} />
            Analysis Results
          </button>
        </div>

        {/* Upload Section */}
        {activeTab === 'upload' && (
          <div className="upload-grid">
            {/* Resume Upload */}
            <div className="card">
              <div className="card-header">
                <FileText size={24} color="#2563eb" />
                <h2 className="card-title">Upload Resume</h2>
              </div>
              
              <label className="upload-area">
                <div className={`upload-box ${resumeFile ? 'upload-success' : ''}`}>
                  {resumeFile ? (
                    <div className="upload-content">
                      <CheckCircle size={48} color="#10b981" />
                      <p className="upload-filename">{resumeFile.name}</p>
                      <p className="upload-hint">Click to change file</p>
                    </div>
                  ) : (
                    <div className="upload-content">
                      <Upload size={48} color="#9ca3af" />
                      <p className="upload-label">Click to upload resume</p>
                      <p className="upload-hint">PDF format only</p>
                    </div>
                  )}
                </div>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleResumeUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </div>

            {/* Job Description Upload */}
            <div className="card">
              <div className="card-header">
                <Briefcase size={24} color="#6366f1" />
                <h2 className="card-title">Job Description</h2>
              </div>
              
              <label className="upload-area" style={{ marginBottom: '16px' }}>
                <div className={`upload-box-small ${jobDescFile ? 'upload-success' : ''}`}>
                  {jobDescFile ? (
                    <div className="upload-content">
                      <CheckCircle size={40} color="#10b981" />
                      <p className="upload-filename-small">{jobDescFile.name}</p>
                    </div>
                  ) : (
                    <div className="upload-content">
                      <Upload size={40} color="#9ca3af" />
                      <p className="upload-label-small">Upload TXT file</p>
                    </div>
                  )}
                </div>
                <input
                  type="file"
                  accept=".txt"
                  onChange={handleJobDescUpload}
                  style={{ display: 'none' }}
                />
              </label>

              <div className="divider">OR</div>

              <textarea
                value={jobDescText}
                onChange={(e) => setJobDescText(e.target.value)}
                placeholder="Paste job description here..."
                className="textarea"
              />
            </div>
          </div>
        )}

        {/* Results Section */}
        {activeTab === 'results' && (
          <div>
            {loading ? (
              <div className="card loading-card">
                <div className="spinner"></div>
                <p className="loading-text">Analyzing your resume...</p>
                <p className="loading-subtext">This may take a few moments</p>
              </div>
            ) : results ? (
              <div className="results-container">
                {/* Score Card */}
                <div className="score-card">
                  <div className="score-content">
                    <div>
                      <h3 className="score-label">Resume-Job Match Score</h3>
                      <div className="score-display">
                        <span className="score-value" style={{ color: getScoreColor(results.match_score) }}>
                          {(results.match_score * 100).toFixed(1)}%
                        </span>
                        <span className="score-assessment">{getScoreLabel(results.match_score)}</span>
                      </div>
                    </div>
                    <Target size={80} style={{ opacity: 0.2 }} color="white" />
                  </div>
                  
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${results.match_score * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Review Report */}
                <div className="card">
                  <div className="review-header">
                    <div className="review-title-container">
                      <Lightbulb size={24} color="#eab308" />
                      <h3 className="review-title">Detailed Analysis</h3>
                    </div>
                    <button onClick={downloadReport} className="download-button">
                      <Download size={16} />
                      Download Report
                    </button>
                  </div>
                  
                  <div className="review-content">
                    {results.review}
                  </div>
                </div>

                {/* Quick Tips */}
                <div className="tips-card">
                  <div className="card-header">
                    <AlertCircle size={24} color="#a855f7" />
                    <h3 className="card-title">Next Steps</h3>
                  </div>
                  <ul className="tips-list">
                    <li className="tip-item">
                      <CheckCircle size={20} color="#a855f7" />
                      <span>Review the strengths and incorporate them in your cover letter</span>
                    </li>
                    <li className="tip-item">
                      <CheckCircle size={20} color="#a855f7" />
                      <span>Address the identified gaps by adding relevant skills or experiences</span>
                    </li>
                    <li className="tip-item">
                      <CheckCircle size={20} color="#a855f7" />
                      <span>Tailor your resume keywords to match the job description</span>
                    </li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="card empty-state">
                <AlertCircle size={64} color="#9ca3af" />
                <p className="empty-title">No results yet</p>
                <p className="empty-subtitle">Upload your files and click Analyze to see results</p>
              </div>
            )}
          </div>
        )}

        {/* Analyze Button */}
        {activeTab === 'upload' && (
          <div className="analyze-container">
            <button
              onClick={analyzeResume}
              disabled={!resumeFile || (!jobDescFile && !jobDescText) || loading}
              className="analyze-button"
            >
              {loading ? 'Analyzing...' : 'Analyze Resume'}
            </button>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Groq AI & Hugging Face • Secure & Private Analysis</p>
      </footer>
    </div>
  );
}