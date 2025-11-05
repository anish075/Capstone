# 🚀 Quick Start Guide

## Step 1: Install Dependencies

### Backend Setup
Open a terminal (Command Prompt) and run:

```cmd
cd backend
pip install -r requirements.txt
```

This will install:
- Flask (web server)
- OpenAI Whisper (AI transcription)
- MoviePy (video processing)
- Noisereduce (audio enhancement)
- And other required libraries

**Note:** The first time you run the backend, Whisper will download the AI model (~150MB for base model).

### Frontend Setup
Open another terminal and run:

```cmd
cd frontend
npm install
```

This will install:
- React and React DOM
- Tailwind CSS (styling)
- Framer Motion (animations)
- Axios (API calls)
- React Dropzone (file upload)

## Step 2: Install FFmpeg

FFmpeg is required for audio/video processing.

### Windows:
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract the zip file
3. Add the `bin` folder to your system PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" in System Variables
   - Add the path to FFmpeg's `bin` folder (e.g., `C:\ffmpeg\bin`)
4. Restart your terminals

### Verify FFmpeg installation:
```cmd
ffmpeg -version
```

## Step 3: Run the Application

### Terminal 1 - Start Backend:
```cmd
cd backend
python app.py
```

You should see:
```
Loading Whisper model...
Whisper model loaded successfully!
🎙️  Audio/Video Transcription Server
Server starting on http://localhost:5000
```

### Terminal 2 - Start Frontend:
```cmd
cd frontend
npm start
```

The browser will automatically open at `http://localhost:3000`

## Step 4: Use the App

1. **Upload a file**: Drag and drop or click to browse
2. **Wait for processing**: Watch the animated loading screen
3. **View transcription**: See the text with timestamps
4. **Copy or download**: Use the action buttons to save your transcription

## 🎯 Supported File Formats

**Audio:**
- MP3
- WAV
- M4A
- FLAC
- OGG

**Video:**
- MP4
- AVI
- MOV
- MKV
- WEBM

## 🐛 Troubleshooting

### "Cannot connect to server"
- Make sure backend is running on port 5000
- Check if `http://localhost:5000/api/health` returns a response

### "FFmpeg not found"
- Verify FFmpeg is in your PATH: `ffmpeg -version`
- Restart your terminal after adding to PATH

### "Module not found" errors
- Reinstall Python packages: `pip install -r requirements.txt`
- Reinstall Node packages: `npm install` (in frontend folder)

### Slow transcription
- The "base" Whisper model is used by default
- For faster processing, edit `backend/app.py` and change to `tiny`
- For better accuracy, change to `small`, `medium`, or `large`

### Port conflicts
Backend uses port 5000, frontend uses port 3000
- If 5000 is taken, edit `backend/app.py` and change the port
- If 3000 is taken, React will prompt you to use another port

## 📊 Performance Tips

- **File Size**: Larger files take longer to process
- **Model Size**: Smaller Whisper models are faster but less accurate
- **First Run**: The first transcription may be slower as Whisper loads

## 🎨 Customization

### Change Whisper Model
Edit `backend/app.py`, line 23:
```python
model = whisper.load_model("base")  # Change to: tiny, small, medium, large
```

### Adjust Noise Reduction
Edit `backend/app.py`, line 51:
```python
reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.8)
# Change prop_decrease: 0.0 (no reduction) to 1.0 (maximum)
```

### Change Colors
Edit `frontend/tailwind.config.js` to customize the color scheme

## ✨ Features

✅ Drag-and-drop file upload
✅ Real-time progress tracking
✅ Automatic noise reduction
✅ AI-powered transcription
✅ Timestamped segments
✅ Copy to clipboard
✅ Download as text file
✅ Futuristic glassmorphism UI
✅ Animated gradients
✅ Responsive design

## 🆘 Need Help?

Check the main README.md for more details or troubleshooting steps.

---

**Ready to transcribe?** 🎙️ Start both servers and open http://localhost:3000
