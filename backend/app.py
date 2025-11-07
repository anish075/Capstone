import os
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
chat_model = None
chat_tokenizer = None
diarization_pipeline = None

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
    """Apply enhanced noise reduction to audio file using noisereduce with optimized settings"""
    global librosa, nr, sf
    try:
        if librosa is None or nr is None or sf is None:
            import librosa as librosa_module
            import noisereduce as nr_module
            import soundfile as sf_module
            librosa = librosa_module
            nr = nr_module
            sf = sf_module
        
        print(f"📥 Loading audio: {audio_path}")
        # Load audio file with original sample rate
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        print("🎯 Applying advanced noise reduction...")
        # Apply noise reduction with stationary + non-stationary noise handling
        # Using two-pass approach for better results
        reduced_noise = nr.reduce_noise(
            y=audio_data, 
            sr=sample_rate,
            stationary=True,           # Handle stationary noise (AC, hum, etc.)
            prop_decrease=1.0,          # Maximum noise reduction
            freq_mask_smooth_hz=500,    # Smooth frequency mask for natural sound
            time_mask_smooth_ms=50,     # Smooth time mask to avoid artifacts
            thresh_n_mult_nonstationary=2,  # Aggressive non-stationary noise reduction
            sigmoid_slope_nonstationary=10,  # Sharper noise gate
            n_std_thresh_stationary=1.5      # Threshold for stationary noise detection
        )
        
        # Save the cleaned audio
        cleaned_path = audio_path.rsplit('.', 1)[0] + '_cleaned.wav'
        sf.write(cleaned_path, reduced_noise, sample_rate)
        print(f"✅ Enhanced noise reduction complete: {cleaned_path}")
        
        return cleaned_path
    except Exception as e:
        print(f"⚠️ Noise reduction failed: {str(e)}")
        print("📋 Traceback:", traceback.format_exc())
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
        
        # Transcribe the audio data directly with automatic language detection
        result = model.transcribe(
            audio_data,
            verbose=False,
            language=None,  # Auto-detect language (supports 99+ languages)
            task='transcribe',  # Use 'translate' to translate to English
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

def perform_simple_speaker_clustering(audio_path):
    """Simple speaker clustering based on audio energy and pitch changes"""
    try:
        global librosa, sf
        if librosa is None or sf is None:
            import librosa as librosa_module
            import soundfile as sf_module
            librosa = librosa_module
            sf = sf_module
        
        print("   Loading audio for clustering...")
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Detect speech segments using energy
        frame_length = 2048
        hop_length = 512
        
        # Calculate energy
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate zero crossing rate (helps detect speaker changes)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate spectral centroid (pitch/voice characteristic)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        features = np.vstack([energy, zcr, spectral_centroid]).T
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Simple clustering based on feature changes
        from sklearn.cluster import KMeans
        
        # Estimate number of speakers (between 2-5)
        n_speakers = min(5, max(2, int(len(features_scaled) / 100)))
        
        print(f"   Estimated {n_speakers} speakers using clustering...")
        
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        # Convert frame indices to timestamps
        times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=hop_length)
        
        # Create diarization segments
        diarization_segments = []
        current_speaker = labels[0]
        start_time = times[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # Speaker change detected
                diarization_segments.append({
                    'start': float(start_time),
                    'end': float(times[i]),
                    'speaker': f'SPEAKER_{current_speaker:02d}'
                })
                current_speaker = labels[i]
                start_time = times[i]
        
        # Add final segment
        diarization_segments.append({
            'start': float(start_time),
            'end': float(times[-1]),
            'speaker': f'SPEAKER_{current_speaker:02d}'
        })
        
        print(f"   ✅ Found {len(set(labels))} unique speakers in {len(diarization_segments)} segments")
        
        return diarization_segments
        
    except Exception as e:
        print(f"   ⚠️ Simple clustering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def perform_speaker_diarization(audio_path, num_speakers=None):
    """Perform speaker diarization using pyannote.audio"""
    global diarization_pipeline
    try:
        print("🎭 Performing speaker diarization...")
        
        # Lazy load the diarization pipeline
        if diarization_pipeline is None:
            try:
                print("   Loading pyannote.audio diarization model...")
                from pyannote.audio import Pipeline
                import torch
                
                # Get HuggingFace token from environment or try without
                hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
                
                if hf_token:
                    print(f"   Using HuggingFace token for authentication...")
                
                # Try to load the pipeline (use 'token' instead of 'use_auth_token' for newer versions)
                try:
                    diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token  # Try old parameter first
                    )
                    print("✅ Diarization model loaded (v3.1)!")
                except TypeError:
                    # New pyannote version uses 'token' instead of 'use_auth_token'
                    try:
                        diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            token=hf_token
                        )
                        print("✅ Diarization model loaded (v3.1 with new token parameter)!")
                    except Exception as e:
                        print(f"   ⚠️ Could not load v3.1: {str(e)}")
                        print("   Trying v3.0...")
                        try:
                            diarization_pipeline = Pipeline.from_pretrained(
                                "pyannote/speaker-diarization-3.0",
                                token=hf_token
                            )
                            print("✅ Diarization model loaded (v3.0)!")
                        except Exception as e2:
                            print(f"   ⚠️ Could not load v3.0: {str(e2)}")
                            print("   Trying v2.1 with revision parameter...")
                            try:
                                diarization_pipeline = Pipeline.from_pretrained(
                                    "pyannote/speaker-diarization",
                                    revision="2.1"
                                )
                                print("✅ Diarization model loaded (v2.1)!")
                            except Exception as e3:
                                print(f"   ⚠️ Could not load v2.1: {str(e3)}")
                                # Use simple clustering as fallback
                                print("   Using simple speaker clustering as fallback...")
                                diarization_pipeline = "simple_clustering"
                
            except Exception as e:
                print(f"⚠️ Could not load any diarization model: {str(e)}")
                print("   Using simple speaker clustering as fallback...")
                import traceback
                traceback.print_exc()
                diarization_pipeline = "simple_clustering"
        
        # Check if we're using simple clustering fallback
        if diarization_pipeline == "simple_clustering":
            print("   ⚠️ Using energy-based speaker clustering (fallback method)")
            return perform_simple_speaker_clustering(audio_path)
        
        # Perform diarization with parameters
        print(f"   Analyzing speakers in: {os.path.basename(audio_path)}")
        
        # Run diarization without constraints first (more reliable)
        try:
            print("   Running diarization (auto-detecting speakers)...")
            diarization = diarization_pipeline(audio_path)
            print("   ✅ Diarization analysis complete")
        except Exception as e:
            print(f"   ⚠️ Diarization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Convert to list of speaker segments
        speaker_segments = []
        unique_speakers = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            unique_speakers.add(speaker)
            print(f"   📍 {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
        
        num_unique_speakers = len(unique_speakers)
        print(f"✅ Diarization complete! Found {num_unique_speakers} unique speakers")
        print(f"   Speakers detected: {sorted(unique_speakers)}")
        print(f"   Total segments: {len(speaker_segments)}")
        
        return speaker_segments
        
    except Exception as e:
        print(f"⚠️ Error in diarization: {str(e)}")
        import traceback
        traceback.print_exc()
        print("   Continuing without speaker diarization...")
        return None

def assign_speakers_to_segments(transcription_segments, speaker_segments):
    """Assign speakers to transcription segments based on time overlap"""
    if not speaker_segments:
        return transcription_segments
    
    try:
        print("🎯 Matching speakers with transcript segments...")
        print(f"   Transcript segments: {len(transcription_segments)}")
        print(f"   Speaker segments: {len(speaker_segments)}")
        
        for i, segment in enumerate(transcription_segments):
            segment_start = segment['start']
            segment_end = segment['end']
            segment_mid = (segment_start + segment_end) / 2
            
            # Find which speaker was talking at the midpoint of this segment
            best_speaker = None
            best_overlap = 0
            
            for speaker_seg in speaker_segments:
                # Check if segment overlaps with speaker time
                overlap_start = max(segment_start, speaker_seg['start'])
                overlap_end = min(segment_end, speaker_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker_seg['speaker']
            
            # Assign speaker or default to "SPEAKER_00"
            if best_speaker:
                segment['speaker'] = best_speaker
                print(f"   Segment {i+1} [{segment_start:.1f}-{segment_end:.1f}s]: {best_speaker} (overlap: {best_overlap:.2f}s)")
            else:
                segment['speaker'] = "SPEAKER_00"
                print(f"   Segment {i+1} [{segment_start:.1f}-{segment_end:.1f}s]: No speaker match (defaulting)")
        
        # Rename speakers to Speaker 1, Speaker 2, etc.
        # Sort speakers by their first appearance
        speaker_first_appearance = {}
        for segment in transcription_segments:
            original_speaker = segment['speaker']
            if original_speaker not in speaker_first_appearance:
                speaker_first_appearance[original_speaker] = segment['start']
        
        # Sort by first appearance time
        sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
        speaker_map = {}
        for idx, (original_speaker, _) in enumerate(sorted_speakers, 1):
            speaker_map[original_speaker] = f"Speaker {idx}"
        
        # Apply the mapping
        for segment in transcription_segments:
            original_speaker = segment['speaker']
            segment['speaker'] = speaker_map[original_speaker]
        
        print(f"✅ Assigned {len(speaker_map)} speakers to segments")
        print(f"   Speaker mapping: {speaker_map}")
        return transcription_segments
        
    except Exception as e:
        print(f"⚠️ Error assigning speakers: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original segments with default speaker
        for segment in transcription_segments:
            if 'speaker' not in segment:
                segment['speaker'] = 'Speaker 1'
        return transcription_segments

def summarize_text(text):
    """Summarize text using Google Gemini AI for high-quality summaries"""
    global gemini_model
    try:
        print("🤖 Generating AI-powered summary with Google Gemini...")
        
        # Quick validation
        if len(text.strip()) < 50:
            return {
                'summary': text,
                'keywords': [],
                'word_count': len(text.split()),
                'summary_ratio': '1/1'
            }
        
        # Get API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise Exception("GEMINI_API_KEY not found in environment variables")
        
        print("🔧 Using Google Gemini 1.5 Flash (v1 API for free tier)...")
        
        # Truncate if text is too long (keep first 6000 words for context)
        words = text.split()
        word_count = len(words)
        if word_count > 6000:
            text = ' '.join(words[:6000]) + "..."
            print(f"   Truncated transcript from {word_count} to 6000 words for processing")
        
        # Create summary prompt
        prompt = f"""Please provide a comprehensive summary of the following transcript. 

Your summary should:
1. Capture the main topics and key points discussed
2. Be concise but informative (3-5 paragraphs)
3. Highlight important details, decisions, or conclusions
4. Also extract 10 important keywords from the content

Transcript:
{text}

Provide your response in this exact format:
SUMMARY:
[Your summary here]

KEYWORDS:
[keyword1, keyword2, keyword3, ...]"""

        print("   Sending transcript to Gemini for summarization...")
        
        # Use REST API v1 endpoint (free-tier compatible)
        import requests
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.9,
                "maxOutputTokens": 800
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        # Extract text from response
        result_text = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        print("✅ Gemini summarization complete!")
        
        # Parse the response
        summary = ""
        keywords = []
        
        if "SUMMARY:" in result_text and "KEYWORDS:" in result_text:
            parts = result_text.split("KEYWORDS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            keywords_text = parts[1].strip()
            # Extract keywords (remove brackets, split by comma)
            keywords_text = keywords_text.replace('[', '').replace(']', '')
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        else:
            # Fallback if format isn't followed perfectly
            summary = result_text
            # Try to extract any list at the end as keywords
            lines = result_text.split('\n')
            for line in lines[-5:]:
                if ',' in line:
                    keywords = [k.strip() for k in line.split(',') if k.strip()]
                    break
        
        print("✅ AI summarization complete!")
        
        return {
            'summary': summary,
            'keywords': keywords[:10],  # Limit to 10 keywords
            'word_count': word_count,
            'summary_ratio': f"{len(summary.split())}/{word_count}"
        }
        
    except Exception as e:
        print(f"⚠️ Gemini summarization failed: {str(e)}")
        print("   Falling back to extractive summarization...")
        traceback.print_exc()
        
        # Fallback to simple extractive method
        from collections import Counter
        import re
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) == 0:
            return {
                'summary': text[:500],
                'keywords': [],
                'word_count': len(text.split()),
                'summary_ratio': '1/1'
            }
        
        # Simple word frequency approach as fallback
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\b[a-z]+\b', text.lower())
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        word_freq = Counter(filtered_words)
        
        # Take first few sentences as summary
        fallback_summary = '. '.join(sentences[:5]) + '.'
        keywords = [word for word, _ in word_freq.most_common(10)]
        
        return {
            'summary': fallback_summary,
            'keywords': keywords,
            'word_count': len(text.split()),
            'summary_ratio': f"{len(fallback_summary.split())}/{len(text.split())}"
        }

def chat_with_transcript(transcript_text, summary_text, question, chat_history=[]):
    """Chat with transcript using Google Gemini (primary) with Mistral-7B fallback"""
    # Try Gemini first
    try:
        return chat_with_gemini(transcript_text, summary_text, question, chat_history)
    except Exception as gemini_error:
        print(f"⚠️ Gemini failed: {str(gemini_error)}")
        print("   Falling back to Mistral-7B...")
        # Fall back to Mistral if Gemini fails
        return chat_with_mistral(transcript_text, summary_text, question, chat_history)

def chat_with_gemini(transcript_text, summary_text, question, chat_history=[]):
    """Chat using Google Gemini 1.5 Flash (fast, smart, free)"""
    global gemini_model
    try:
        print(f"💬 Processing question with Gemini: {question}...")
        
        # Get API key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise Exception("GEMINI_API_KEY not found in environment variables")
        
        print("🤖 Using Google Gemini 1.5 Flash (v1 API for free tier)...")
        
        # Prepare context
        max_context_length = 8000  # Gemini handles more context
        context = ""
        
        if summary_text:
            context = f"Summary: {summary_text}\n\n"
        
        # Add transcript (truncated if needed)
        if len(transcript_text) > max_context_length:
            context += f"Transcript excerpt: {transcript_text[:max_context_length]}..."
        else:
            context += f"Full transcript: {transcript_text}"
        
        # Build conversation history
        history_text = ""
        for msg in chat_history[-3:]:  # Last 3 exchanges
            if msg.get('question') and msg.get('answer'):
                history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant analyzing a transcript. Answer questions based on the transcript content.

Transcript Context:
{context}

Previous conversation:
{history_text}

Current question: {question}

Please provide a clear, accurate answer based on the transcript. If the information isn't in the transcript, say so politely. Keep your answer concise and helpful."""
        
        print(f"   Generating response with Gemini 1.5 Flash...")
        
        # Use REST API v1 endpoint (free-tier compatible)
        import requests
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.95,
                "maxOutputTokens": 500
            }
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        answer = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        
        if not answer or len(answer) < 10:
            answer = "I apologize, but I couldn't generate a proper response. Could you rephrase your question?"
        
        print("✅ Response generated with Gemini!")
        
        return {
            'answer': answer,
            'context_used': len(context),
            'model': 'Gemini 1.5 Flash'
        }
        
    except Exception as e:
        print(f"⚠️ Error in Gemini chat: {str(e)}")
        import traceback
        traceback.print_exc()
        # Re-raise to trigger fallback
        raise

def chat_with_mistral(transcript_text, summary_text, question, chat_history=[]):
    """Chat with transcript using Mistral-7B-Instruct for high-quality responses"""
    global chat_model, chat_tokenizer
    try:
        print(f"💬 Processing question: {question}...")
        
        # Lazy load the chat model (Mistral-7B-Instruct)
        if chat_model is None:
            print("🤖 Loading Mistral-7B-Instruct-v0.2 model...")
            print("⏳ First-time setup: downloading ~15GB model (this may take 10-20 minutes)...")
            print("   This is a one-time download. Subsequent uses will be instant!")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            
            try:
                print("   Downloading tokenizer...")
                chat_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                print("   Downloading model (this will take a while)...")
                chat_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                print("✅ Mistral-7B model loaded successfully!")
                
            except Exception as model_error:
                print(f"⚠️ Could not load Mistral-7B model: {str(model_error)}")
                print("   Falling back to rule-based assistant...")
                # Fall back to rule-based system
                return chat_with_transcript_fallback(transcript_text, summary_text, question, chat_history)
        
        # Prepare context from transcript
        import re
        
        # Limit context to most relevant parts
        max_context_length = 1500
        context = ""
        
        if summary_text:
            context = f"Summary: {summary_text}\n\n"
        
        # Add transcript (truncated if needed)
        if len(transcript_text) > max_context_length:
            context += f"Transcript excerpt: {transcript_text[:max_context_length]}..."
        else:
            context += f"Full transcript: {transcript_text}"
        
        # Build conversation history
        history_text = ""
        for msg in chat_history[-3:]:  # Last 3 exchanges for better context
            if msg.get('question') and msg.get('answer'):
                history_text += f"User: {msg['question']}\nAssistant: {msg['answer']}\n\n"
        
        # Create prompt using Mistral's instruction format
        # Mistral uses [INST] tags for instructions
        prompt = f"""[INST] You are a helpful AI assistant analyzing a transcript. Answer questions based on the transcript content provided below.

Transcript Context:
{context}

Previous conversation:
{history_text}

Current question: {question}

Please provide a clear, accurate answer based on the transcript. If the information isn't in the transcript, say so politely. [/INST]"""
        
        print(f"   Generating response with Mistral-7B...")
        
        # Tokenize input
        inputs = chat_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        # Generate response with better parameters for Mistral
        import torch
        with torch.no_grad():
            outputs = chat_model.generate(
                inputs.input_ids,
                max_new_tokens=200,  # Allow longer responses
                temperature=0.7,
                top_p=0.95,  # Increased for more diverse responses
                do_sample=True,
                pad_token_id=chat_tokenizer.eos_token_id,
                eos_token_id=chat_tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )
        
        # Decode response
        full_response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part (after [/INST])
        if "[/INST]" in full_response:
            answer = full_response.split("[/INST]")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        # Clean up response
        # Remove any remaining instruction tags
        answer = answer.replace("[INST]", "").replace("[/INST]", "").strip()
        
        # Take first coherent paragraph
        paragraphs = answer.split("\n\n")
        answer = paragraphs[0] if paragraphs else answer
        
        # Stop at next question if it appears
        if "User:" in answer:
            answer = answer.split("User:")[0]
        if "Question:" in answer:
            answer = answer.split("Question:")[0]
        
        answer = answer.strip()
        
        if not answer or len(answer) < 10:
            answer = "I apologize, but I couldn't generate a proper response. Could you rephrase your question?"
        
        print("✅ Response generated with Mistral-7B!")
        
        return {
            'answer': answer,
            'context_used': len(context),
            'model': 'Mistral-7B-Instruct-v0.2'
        }
        
    except Exception as e:
        print(f"⚠️ Error in Mistral-7B chat: {str(e)}")
        import traceback
        traceback.print_exc()
        print("   Falling back to rule-based assistant...")
        # Fall back to rule-based system on any error
        return chat_with_transcript_fallback(transcript_text, summary_text, question, chat_history)


def chat_with_transcript_fallback(transcript_text, summary_text, question, chat_history=[]):
    """Fallback rule-based chat system (used when LLM fails)"""
    try:
        print(f"💬 Using fallback assistant for: {question}...")
        
        import re
        from collections import Counter
        
        question_lower = question.lower()
        transcript_lower = transcript_text.lower()
        
        # Enhanced question classification with more patterns
        is_summary = any(word in question_lower for word in ['summarize', 'summary', 'main point', 'key point', 'overview', 'gist', 'recap'])
        is_about = any(word in question_lower for word in ['about', 'regarding', 'concerning', 'discuss'])
        is_mention = any(word in question_lower for word in ['mention', 'say', 'talk', 'speak'])
        is_happen = any(word in question_lower for word in ['happen', 'occur', 'going on', 'taking place'])
        is_explain = any(word in question_lower for word in ['explain', 'describe', 'detail'])
        
        # Question type detection
        is_what = 'what' in question_lower
        is_who = 'who' in question_lower
        is_when = 'when' in question_lower
        is_where = 'where' in question_lower
        is_why = 'why' in question_lower
        is_how = 'how' in question_lower
        
        # If asking for summary, return the summary
        if is_summary:
            if summary_text:
                answer = f"Here's a summary of the transcript:\n\n{summary_text}\n\nWould you like me to elaborate on any specific part?"
            else:
                # Generate quick summary from transcript
                sentences = re.split(r'[.!?]+', transcript_text)
                sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
                answer = f"The transcript covers: {'. '.join(sentences[:3])}."
            
            print("✅ Response generated (summary)!")
            return {
                'answer': answer,
                'context_used': len(answer),
                'model': 'Enhanced Assistant'
            }
        
        # Extract keywords from question (excluding common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'what', 'when',
                     'where', 'who', 'why', 'how', 'can', 'you', 'about', 'tell', 'me', 'this',
                     'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'could',
                     'would', 'should', 'will', 'just', 'like', 'going'}
        
        question_words = [w for w in re.findall(r'\b[a-z]+\b', question_lower) 
                         if w not in stop_words and len(w) > 2]
        
        # If no keywords, try to extract key phrases
        if not question_words:
            # Look for quoted text or proper nouns
            quoted = re.findall(r'"([^"]+)"', question)
            if quoted:
                question_words = [quoted[0].lower()]
            else:
                answer = "I'd be happy to help! Could you rephrase your question to be more specific?"
                print("✅ Response generated (clarification)!")
                return {
                    'answer': answer,
                    'context_used': 0,
                    'model': 'Enhanced Assistant'
                }
        
        print(f"   Keywords: {', '.join(question_words)}")
        
        # Find relevant sentences with flexible matching
        sentences = re.split(r'[.!?]+', transcript_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Score based on keyword matches and context
            score = 0
            for word in question_words:
                if word in sentence_lower:
                    score += 2
                # Also check for partial matches (word stems)
                elif any(word[:4] in w or w[:4] in word for w in sentence_lower.split() if len(w) > 4):
                    score += 1
            
            if score > 0:
                relevant_sentences.append((score, sentence))
        
        # Sort by relevance
        relevant_sentences.sort(reverse=True, key=lambda x: x[0])
        
        print(f"   Found {len(relevant_sentences)} relevant sentences")
        
        # Build answer from most relevant sentences
        if relevant_sentences:
            # Take top sentences based on score
            top_sentences = [s[1] for s in relevant_sentences[:4]]
            
            # Create contextual response based on question type
            if is_what:
                if is_about or is_mention:
                    answer = f"Regarding that topic, the transcript states: \"{top_sentences[0]}\""
                elif is_happen or is_explain:
                    answer = f"Here's what happened: {top_sentences[0]}"
                else:
                    answer = f"Based on the transcript: {top_sentences[0]}"
            
            elif is_who:
                answer = f"According to the transcript: {top_sentences[0]}"
            
            elif is_when:
                answer = f"The transcript mentions: {top_sentences[0]}"
            
            elif is_where:
                answer = f"From the transcript: {top_sentences[0]}"
            
            elif is_why:
                answer = f"The reason given is: {top_sentences[0]}"
            
            elif is_how:
                answer = f"Here's how it was described: {top_sentences[0]}"
            
            else:
                # Generic response
                answer = f"{top_sentences[0]}"
            
            # Add additional context if available and relevant
            if len(top_sentences) > 1 and len(answer) < 300:
                second_sentence = top_sentences[1]
                # Make it flow naturally
                if not second_sentence[0].isupper():
                    second_sentence = second_sentence[0].upper() + second_sentence[1:]
                answer += f"\n\nAdditionally: {second_sentence}"
            
            # Add more context if question seems to want more detail
            if any(word in question_lower for word in ['explain', 'detail', 'more', 'elaborate']) and len(top_sentences) > 2:
                answer += f"\n\nFurthermore: {top_sentences[2]}"
            
            print("✅ Response generated (context-based)!")
            return {
                'answer': answer,
                'context_used': len(answer),
                'model': 'Enhanced Assistant'
            }
        
        else:
            # No direct match - provide helpful fallback
            answer = f"I couldn't find specific information about '{' '.join(question_words[:3])}' in the transcript. "
            
            # Try to provide related information
            if summary_text:
                answer += f"\n\nHowever, here's what the transcript covers:\n{summary_text[:300]}"
                if len(summary_text) > 300:
                    answer += "..."
            else:
                # Show first few sentences as context
                sentences = re.split(r'[.!?]+', transcript_text)
                first_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
                if first_sentences:
                    answer += f"\n\nThe transcript discusses: {'. '.join(first_sentences)}."
            
            answer += "\n\nCould you rephrase your question or ask about a different aspect?"
            
            print("✅ Response generated (no match, with context)!")
            return {
                'answer': answer,
                'context_used': len(answer),
                'model': 'Enhanced Assistant'
            }
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        raise


        assistant_marker = "<|assistant|>"
        if assistant_marker in full_response:
            answer = full_response.split(assistant_marker)[-1].strip()
        else:
            answer = full_response.split("</s>")[-1].strip()
        
        # Clean up the response
        answer = answer.split("<|")[0].strip()  # Remove any following special tokens
        answer = answer.split("User:")[0].strip()  # Stop at next user input
        
        if not answer:
            answer = "I'm not sure I can answer that based on the transcript provided."
        
        print("✅ Response generated!")
        
        return {
            'answer': answer,
            'context_used': len(context),
            'model': 'TinyLlama-1.1B-Chat'
        }
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
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
        
        # Perform speaker diarization
        print(f"🎭 Starting speaker diarization...")
        speaker_segments = perform_speaker_diarization(cleaned_audio_path)
        
        # Assign speakers to transcription segments
        if speaker_segments:
            transcription_result['segments'] = assign_speakers_to_segments(
                transcription_result['segments'], 
                speaker_segments
            )
            print(f"✅ Speaker diarization complete!")
        else:
            # Add default speaker to all segments
            for segment in transcription_result['segments']:
                segment['speaker'] = 'Speaker 1'
            print(f"⚠️ Using single speaker fallback")
        
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

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with transcript using local LLM"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        question = data.get('question', '').strip()
        transcript = data.get('transcript', '').strip()
        summary = data.get('summary', '').strip()
        chat_history = data.get('chat_history', [])
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not transcript:
            return jsonify({'error': 'No transcript provided'}), 400
        
        print("💬 Starting chat interaction...")
        print(f"   Question: {question[:100]}...")
        
        # Generate response
        chat_result = chat_with_transcript(transcript, summary, question, chat_history)
        
        print(f"✅ Chat response generated!")
        
        return jsonify({
            'success': True,
            'answer': chat_result['answer'],
            'model': chat_result['model'],
            'context_used': chat_result['context_used']
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing chat request: {error_msg}")
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
