import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import LoadingAnimation from './components/LoadingAnimation';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [transcription, setTranscription] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTranscriptionComplete = (data) => {
    setTranscription(data);
    setLoading(false);
    setError(null);
  };

  const handleTranscriptionStart = () => {
    setLoading(true);
    setError(null);
    setTranscription(null);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setLoading(false);
  };

  const handleReset = () => {
    setTranscription(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="min-h-screen animated-gradient overflow-hidden">
      {/* Animated background blobs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float"></div>
        <div className="absolute top-1/3 -right-20 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute bottom-1/4 left-1/3 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-float" style={{ animationDelay: '4s' }}></div>
      </div>

      {/* Main content */}
      <div className="relative z-10 container mx-auto px-4 py-8 md:py-12">
        {/* Header */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-5xl md:text-7xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
            🎙️ Audio Transcriber
          </h1>
          <p className="text-gray-300 text-lg md:text-xl max-w-2xl mx-auto mb-4">
            Upload your audio or video files for instant transcription with advanced AI noise reduction
          </p>
          
          {/* Feature badges */}
          <div className="flex flex-wrap justify-center gap-3 mt-6">
            <span className="px-4 py-2 glass rounded-full text-sm font-medium text-purple-300 flex items-center gap-2">
              🤖 Whisper AI
            </span>
            <span className="px-4 py-2 glass rounded-full text-sm font-medium text-blue-300 flex items-center gap-2">
              🎭 Speaker Diarization
            </span>
            <span className="px-4 py-2 glass rounded-full text-sm font-medium text-pink-300 flex items-center gap-2">
              🌍 99+ Languages
            </span>
            <span className="px-4 py-2 glass rounded-full text-sm font-medium text-green-300 flex items-center gap-2">
              🧠 AI Chat & Summary
            </span>
          </div>
        </motion.div>

        {/* Main content area */}
        <div className="max-w-6xl mx-auto">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <LoadingAnimation />
              </motion.div>
            ) : transcription ? (
              <motion.div
                key="transcription"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <TranscriptionDisplay 
                  transcription={transcription} 
                  onReset={handleReset}
                />
              </motion.div>
            ) : (
              <motion.div
                key="upload"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3 }}
              >
                <FileUpload 
                  onTranscriptionComplete={handleTranscriptionComplete}
                  onTranscriptionStart={handleTranscriptionStart}
                  onError={handleError}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 glass-strong rounded-2xl p-6 border-red-500/50"
            >
              <div className="flex items-center gap-3">
                <span className="text-3xl">⚠️</span>
                <div>
                  <h3 className="text-red-400 font-semibold text-lg">Error</h3>
                  <p className="text-gray-300 mt-1">{error}</p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="mt-4 px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold hover:from-purple-500 hover:to-blue-500 transition-all duration-300"
              >
                Try Again
              </button>
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <motion.div 
          className="text-center mt-16 text-gray-400 text-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
        >
          <p>Powered by OpenAI Whisper • Built with React & Flask</p>
          <p className="mt-2">Supports MP3, WAV, MP4, AVI, MOV, and more</p>
        </motion.div>
      </div>
    </div>
  );
}

export default App;
