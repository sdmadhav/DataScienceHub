# 🚀 AI Resume Analyzer - Full Stack Deployment Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Environment Configuration](#environment-configuration)
5. [Running the Application](#running-the-application)
6. [Deployment Options](#deployment-options)
7. [Features](#features)

---

## Prerequisites

### Required Software
- **Python 3.8+** (for backend)
- **Node.js 16+** and npm (for frontend)
- **Git** (for version control)

### API Keys
1. **Hugging Face API Key**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Groq API Key**: Get from [console.groq.com](https://console.groq.com)

---

## Backend Setup

### 1. Create Project Directory
```bash
mkdir ai-resume-analyzer
cd ai-resume-analyzer
mkdir backend
cd backend
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn python-multipart pdfplumber huggingface-hub groq python-dotenv numpy
```

### 4. Create `requirements.txt`
```bash
pip freeze > requirements.txt
```

### 5. Create `.env` File
```bash
# Create .env file in backend directory
echo "HF_API_KEY=your_huggingface_api_key_here" > .env
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
```

### 6. Save Backend Code
- Copy the backend code (from artifact) and save as `main.py`

---

## Frontend Setup

### 1. Create React App
```bash
cd ..  # Go back to project root
npx create-react-app frontend
cd frontend
```

### 2. Install Dependencies
```bash
npm install lucide-react
```

### 3. Setup Tailwind CSS
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### 4. Configure Tailwind
Edit `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Edit `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 5. Replace App Component
- Replace content of `src/App.js` with the frontend code (from artifact)

---

## Environment Configuration

### Backend `.env` File
```env
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Frontend Configuration
If deploying to production, update the API URL in the frontend code:
```javascript
// Change this line in App.js
const response = await fetch('http://localhost:8000/analyze', {
// To your production backend URL:
const response = await fetch('https://your-backend-url.com/analyze', {
```

---

## Running the Application

### Start Backend Server
```bash
cd backend
# Activate virtual environment first
python main.py

# Or with uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will run on: `http://localhost:8000`

### Start Frontend Development Server
```bash
cd frontend
npm start
```

Frontend will run on: `http://localhost:3000`

---

## Deployment Options

### Option 1: Deploy Backend to Railway.app

1. **Create account** at [railway.app](https://railway.app)
2. **Create new project** → Deploy from GitHub
3. **Add environment variables** (HF_API_KEY, GROQ_API_KEY)
4. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Deploy** and get your backend URL

### Option 2: Deploy Backend to Render.com

1. **Create account** at [render.com](https://render.com)
2. **New Web Service** → Connect GitHub repo
3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Add environment variables**
5. **Deploy**

### Option 3: Deploy Frontend to Vercel

1. **Create account** at [vercel.com](https://vercel.com)
2. **Import project** from GitHub
3. **Configure**:
   - Framework Preset: Create React App
   - Build Command: `npm run build`
   - Output Directory: `build`
4. **Add environment variable** for backend URL
5. **Deploy**

### Option 4: Deploy Frontend to Netlify

1. **Create account** at [netlify.com](https://netlify.com)
2. **New site from Git**
3. **Configure**:
   - Build Command: `npm run build`
   - Publish Directory: `build`
4. **Deploy**

---

## Features

### ✅ Core Features
- **PDF Resume Upload**: Upload and parse PDF resumes
- **Job Description Input**: File upload or text paste
- **AI-Powered Analysis**: Using Groq LLM for intelligent review
- **Match Score**: Cosine similarity between resume and JD
- **Keyword Extraction**: Identify common and missing keywords

### ✨ Enhanced Features
- **Beautiful UI**: Modern, gradient-based design
- **Real-time Analysis**: Loading states and progress indicators
- **Detailed Reports**: Strengths, weaknesses, ATS tips, improvements
- **Download Report**: Export analysis as text file
- **Multi-tab Interface**: Organized workflow
- **Responsive Design**: Works on desktop and mobile

### 🎯 Advanced Enhancements You Can Add

1. **User Authentication**
   ```bash
   pip install python-jose[cryptography] passlib[bcrypt]
   ```

2. **Database Storage** (SQLite/PostgreSQL)
   ```bash
   pip install sqlalchemy psycopg2-binary
   ```

3. **Rate Limiting**
   ```bash
   pip install slowapi
   ```

4. **File Storage** (AWS S3)
   ```bash
   pip install boto3
   ```

5. **Email Reports** (SendGrid)
   ```bash
   pip install sendgrid
   ```

6. **Analytics Dashboard**
   - Track usage metrics
   - Visualize match scores over time
   - A/B testing for resume versions

---

## Testing the Application

### Test Backend Endpoints

```bash
# Health check
curl http://localhost:8000/

# Test analyze endpoint (requires form-data)
curl -X POST http://localhost:8000/analyze \
  -F "resume=@/path/to/resume.pdf" \
  -F "job_description_text=Software Engineer position..."
```

### Test Frontend
1. Open `http://localhost:3000`
2. Upload a sample PDF resume
3. Paste or upload a job description
4. Click "Analyze Resume"
5. View results and download report

---

## Troubleshooting

### Common Issues

**1. CORS Errors**
- Ensure backend CORS is configured correctly
- Check frontend is using correct backend URL

**2. API Key Errors**
- Verify `.env` file exists and contains valid keys
- Restart backend after updating `.env`

**3. PDF Parsing Fails**
- Check PDF is not encrypted or password-protected
- Ensure PDF contains selectable text (not scanned images)

**4. Port Already in Use**
```bash
# Kill process on port 8000 (backend)
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill -9
```

---

## Project Structure

```
ai-resume-analyzer/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   └── venv/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── index.css
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

---

## Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com
- **React Docs**: https://react.dev
- **Tailwind CSS**: https://tailwindcss.com
- **Groq API**: https://console.groq.com/docs
- **Hugging Face**: https://huggingface.co/docs

---

## License
MIT License - Feel free to use this project for personal or commercial purposes.

## Support
For issues or questions, create an issue in the GitHub repository.

---

**Happy Analyzing! 🎉**
