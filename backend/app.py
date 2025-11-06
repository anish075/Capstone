import os
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Set FFmpeg path for MoviePy and Whisper
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
    
    # Add FFmpeg directory to PATH so Whisper can find it
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    if ffmpeg_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    
    print(f"✅ FFmpeg configured: {ffmpeg_path}")
except Exception as e:
    print(f"⚠️ Warning: Could not configure FFmpeg: {e}")

# Lazy imports to avoid startup issues
whisper = None
nr = None
librosa = None
sf = None
VideoFileClip = None
summarizer = None
transformers = None

app = Flask(__name__)
CORS(app)

# Configuration
# Use absolute path for uploads folder to avoid path issues
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(SCRIPT_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'flac', 'ogg'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"📁 Upload folder: {UPLOAD_FOLDER}")

# Global model variable
model = None

def load_dependencies():
    """Lazy load dependencies"""
    global whisper, nr, librosa, sf, VideoFileClip, model
    
    if model is not None:
        return
    
    try:
        print("Loading dependencies...")
        import whisper as whisper_module
        import noisereduce as nr_module
        import librosa as librosa_module
        import soundfile as sf_module
        from moviepy.editor import VideoFileClip as VideoFileClip_class
        
        whisper = whisper_module
        nr = nr_module
        librosa = librosa_module
        sf = sf_module
        VideoFileClip = VideoFileClip_class
        
        print("Loading Whisper model (base)...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
    except Exception as e:
        print(f"Error loading dependencies: {e}")
        traceback.print_exc()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_from_video(video_path):
    """Extract audio from video file using FFmpeg directly"""
    try:
        import subprocess
        import imageio_ffmpeg
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Using FFmpeg: {ffmpeg_exe}")
        
        audio_path = video_path.rsplit('.', 1)[0] + '_extracted.wav'
        print(f"Extracting audio from: {video_path}")
        print(f"Output will be: {audio_path}")
        
        # Use FFmpeg directly with subprocess (more reliable on Windows)
        cmd = [
            ffmpeg_exe,
            '-i', video_path,           # Input file
            '-vn',                       # No video
            '-acodec', 'pcm_s16le',     # Audio codec
            '-ar', '16000',              # Sample rate (Whisper likes 16kHz)
            '-ac', '1',                  # Mono audio
            '-y',                        # Overwrite output file
            audio_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run FFmpeg with proper Windows subprocess handling
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            text=True
        )
        
        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            raise Exception(f"FFmpeg failed with return code {result.returncode}")
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file was not created: {audio_path}")
        
        print(f"✅ Audio extracted successfully: {os.path.getsize(audio_path)} bytes")
        return audio_path
        
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        traceback.print_exc()
        raise

def apply_noise_reduction(audio_path):
    """Apply noise reduction to audio file"""
    global librosa, nr, sf
    try:
        if librosa is None or nr is None or sf is None:
            import librosa as librosa_module
            import noisereduce as nr_module
            import soundfile as sf_module
            librosa = librosa_module
            nr = nr_module
            sf = sf_module
        
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.8)
        
        # Save the cleaned audio
        cleaned_path = audio_path.rsplit('.', 1)[0] + '_cleaned.wav'
        sf.write(cleaned_path, reduced_noise, sample_rate)
        
        return cleaned_path
    except Exception as e:
        print(f"Error in noise reduction: {str(e)}")
        # If noise reduction fails, return original audio
        return audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    global model, whisper, librosa
    try:
        # Ensure model is loaded
        if model is None:
            if whisper is None:
                import whisper as whisper_module
                whisper = whisper_module
            print("Loading Whisper model (base)...")
            model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
        
        # Load audio with librosa instead of letting Whisper use FFmpeg
        if librosa is None:
            import librosa as librosa_module
            librosa = librosa_module
        
        # Load audio as numpy array (Whisper expects 16kHz)
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Transcribe the audio data directly
        result = model.transcribe(
            audio_data,
            verbose=False,
            language='en',  # Set to None for auto-detection
            task='transcribe',
            word_timestamps=True
        )
        
        # Format the result with timestamps
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return {
            'text': result['text'].strip(),
            'segments': segments,
            'language': result.get('language', 'unknown')
        }
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        raise

def summarize_text(text):
    """Summarize text using extractive summarization (no external models needed)"""
    try:
        from collections import Counter
        import re
        
        print("🧠 Generating extractive summary...")
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) == 0:
            return {
                'summary': text[:500],
                'keywords': [],
                'word_count': len(text.split()),
                'summary_ratio': '1/1'
            }
        
        # Score sentences based on word frequency (TF-IDF-like approach)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                     'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'just',
                     'like', 'so', 'well', 'very', 'really', 'also', 'now', 'then', 'there'}
        
        # Count word frequencies
        words = re.findall(r'\b[a-z]+\b', text.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(filtered_words)
        
        # Score each sentence
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_words = re.findall(r'\b[a-z]+\b', sentence.lower())
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
            # Normalize by sentence length
            if len(sentence_words) > 0:
                sentence_scores[i] = score / len(sentence_words)
        
        # Select top sentences (aim for 20-30% of original)
        num_summary_sentences = max(3, min(10, len(sentences) // 4))
        top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_summary_sentences]
        
        # Keep original order
        top_sentence_indices.sort()
        summary_sentences = [sentences[i] for i in top_sentence_indices]
        combined_summary = '. '.join(summary_sentences)
        
        # Ensure proper ending
        if combined_summary and combined_summary[-1] not in '.!?':
            combined_summary += '.'
        
        # Extract keywords (top 10 most frequent meaningful words)
        keywords = [word for word, _ in word_freq.most_common(10)]
        
        print("✅ Summarization complete!")
        
        return {
            'summary': combined_summary,
            'keywords': keywords,
            'word_count': len(text.split()),
            'summary_ratio': f"{len(combined_summary.split())}/{len(text.split())}"
        }
        
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        traceback.print_exc()
        raise

@app.route('/', methods=['GET'])
def home():
    """Home page with server info"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Transcription API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 40px;
                max-width: 600px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }
            h1 { margin-top: 0; font-size: 2.5em; }
            .status { 
                background: #10b981; 
                padding: 10px 20px; 
                border-radius: 10px; 
                display: inline-block;
                margin: 20px 0;
            }
            .endpoint {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
            }
            .method {
                background: #3b82f6;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 0.8em;
                margin-right: 10px;
            }
            a {
                color: #60a5fa;
                text-decoration: none;
            }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Audio Transcription API</h1>
            <div class="status">✅ Server Running</div>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <strong>/api/health</strong>
                <p>Health check endpoint</p>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <strong>/api/transcribe</strong>
                <p>Upload audio/video file for transcription</p>
            </div>
            
            <h2>Frontend:</h2>
            <p>Access the web interface at: <a href="http://localhost:3000" target="_blank">http://localhost:3000</a></p>
            
            <h2>Features:</h2>
            <ul>
                <li>Audio/Video file upload</li>
                <li>AI-powered noise reduction</li>
                <li>OpenAI Whisper transcription</li>
                <li>Timestamp generation</li>
            </ul>
            
            <p style="margin-top: 40px; opacity: 0.8; font-size: 0.9em;">
                Powered by Flask, Whisper AI, and Python
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Main transcription endpoint"""
    temp_files = []
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"💾 Saving file to: {file_path}")
        file.save(file_path)
        temp_files.append(file_path)
        
        # Verify file was saved
        if not os.path.exists(file_path):
            raise Exception(f"File was not saved correctly: {file_path}")
        print(f"✅ File saved successfully: {os.path.getsize(file_path)} bytes")
        
        # Determine if it's a video file
        file_ext = filename.rsplit('.', 1)[1].lower()
        is_video = file_ext in {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        
        audio_path = file_path
        
        # Extract audio if video
        if is_video:
            print(f"🎬 Extracting audio from video: {filename}")
            audio_path = extract_audio_from_video(file_path)
            temp_files.append(audio_path)
            print(f"✅ Audio extracted to: {audio_path}")
        
        # Apply noise reduction
        print(f"🔇 Applying noise reduction to: {audio_path}")
        cleaned_audio_path = apply_noise_reduction(audio_path)
        if cleaned_audio_path != audio_path:
            temp_files.append(cleaned_audio_path)
        print(f"✅ Noise reduction complete: {cleaned_audio_path}")
        
        # Transcribe audio
        print(f"🤖 Transcribing audio: {cleaned_audio_path}")
        transcription_result = transcribe_audio(cleaned_audio_path)
        print(f"✅ Transcription complete!")
        
        return jsonify({
            'success': True,
            'transcription': transcription_result['text'],
            'segments': transcription_result['segments'],
            'language': transcription_result['language'],
            'filename': filename
        })
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error deleting temp file {temp_file}: {str(e)}")

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Summarize transcript text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        print("🧠 Starting summarization...")
        print(f"   Input length: {len(text.split())} words")
        
        # Summarize the text
        summary_result = summarize_text(text)
        
        print(f"✅ Summary generated: {summary_result['summary_ratio']} words")
        
        return jsonify({
            'success': True,
            'summary': summary_result['summary'],
            'keywords': summary_result['keywords'],
            'word_count': summary_result['word_count'],
            'summary_ratio': summary_result['summary_ratio']
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing summarization request: {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({'message': 'API is working!'})

if __name__ == '__main__':
    print("=" * 50)
    print("🎙️  Audio/Video Transcription Server")
    print("=" * 50)
    print("Server starting on http://localhost:5000")
    print("Whisper model: base")
    print("Press CTRL+C to stop the server")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
