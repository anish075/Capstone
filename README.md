# 🎙️ Audio/Video Transcription App

A full-stack web application that transcribes audio and video files with noise reduction, built with React, Tailwind CSS, Flask, and OpenAI Whisper.

## ✨ Features

- 🎬 **Video/Audio Upload**: Drag-and-drop or click to upload
- 🔇 **Noise Reduction**: Automatic noise reduction before transcription
- 🗣️ **Speech-to-Text**: Powered by OpenAI Whisper (runs locally)
- 🌍 **Multilingual Support**: Automatic detection and transcription of 99+ languages
- 🎭 **Speaker Diarization**: Identifies and labels different speakers (Speaker 1, Speaker 2, etc.)
- ⏱️ **Timestamps**: View transcription with accurate timestamps
- 🧠 **AI Chat Assistant**: Chat with your transcript using Phi-2 LLM
- 📝 **AI Summarization**: Get extractive summaries with key topics
- 📋 **Copy/Download**: Easy copy to clipboard or download as text file
- 🎨 **Futuristic UI**: Glassmorphism design with animated gradients

## 🚀 Tech Stack

**Frontend:**
- React 18
- Tailwind CSS
- Axios for API calls
- Framer Motion for animations

**Backend:**
- Flask (Python)
- OpenAI Whisper (multilingual speech recognition)
- Phi-2 (Microsoft 2.7B parameter LLM for chat)
- Pyannote.audio (speaker diarization)
- Transformers (Hugging Face)
- MoviePy (video processing)
- Noisereduce (audio enhancement)
- Librosa (audio loading)
- FFmpeg

## 📋 Prerequisites

- Node.js (v16 or higher)
- Python 3.8 or higher
- FFmpeg installed and added to PATH

### Installing FFmpeg

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract and add to system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## 🛠️ Installation

### 1. Clone/Navigate to the project

```bash
cd "c:\Users\anish\OneDrive\Desktop\Capstone\Anish"
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

## ▶️ Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
python app.py
```

Backend will run on `http://localhost:5000`

### Start Frontend (Terminal 2)

```bash
cd frontend
npm start
```

Frontend will run on `http://localhost:3000`

## 🎯 Usage

1. Open your browser to `http://localhost:3000`
2. Drag and drop or click to upload an audio/video file
3. Wait for processing (noise reduction + transcription + speaker diarization)
4. View transcription with speaker labels, timestamps, and detected language
5. **Summarize**: Click to generate AI summary with key topics
6. **Chat**: Use the floating chat assistant to ask questions about the transcript
7. Copy to clipboard or download as text file

### Multilingual Usage

The app **automatically detects** the language being spoken:
- No configuration needed
- Supports 99+ languages (English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi, etc.)
- Language badge shown in UI
- See [MULTILINGUAL_SUPPORT.md](MULTILINGUAL_SUPPORT.md) for detailed documentation

## 📁 Project Structure

```
Anish/
├── backend/
│   ├── app.py                 # Flask server
│   ├── requirements.txt       # Python dependencies
│   └── uploads/              # Temporary upload folder
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.jsx
│   │   │   ├── TranscriptionDisplay.jsx
│   │   │   └── LoadingAnimation.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── index.js
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## 🎨 UI Features

- **Glassmorphism**: Frosted glass effect with backdrop blur
- **Animated Gradients**: Dynamic background animations
- **Dark Mode**: Sleek dark interface
- **Responsive**: Works on desktop and mobile
- **Progress Indicators**: Visual feedback during processing

## 📝 Supported Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Video**: MP4, AVI, MOV, MKV, WEBM

## ⚙️ Configuration

The Whisper model size can be changed in `backend/app.py`:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

## 🐛 Troubleshooting

**Issue**: ModuleNotFoundError
- **Solution**: Ensure all Python packages are installed: `pip install -r requirements.txt`

**Issue**: FFmpeg not found
- **Solution**: Install FFmpeg and add to system PATH

**Issue**: CORS errors
- **Solution**: Ensure backend is running on port 5000

## 📄 License

MIT License - feel free to use this project for learning or commercial purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---


