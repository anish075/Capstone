import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { motion } from 'framer-motion';

const FileUpload = ({ onTranscriptionComplete, onTranscriptionStart, onError }) => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [currentStatus, setCurrentStatus] = useState('');

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const maxSize = 500 * 1024 * 1024; // 500MB

    if (file.size > maxSize) {
      onError('File size exceeds 500MB limit');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    onTranscriptionStart();
    setUploadProgress(0);
    setCurrentStatus('Uploading file...');

    try {
      const response = await axios.post('http://localhost:5000/api/transcribe', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
          if (progress === 100) {
            setCurrentStatus('Processing audio...');
          }
        },
      });

      if (response.data.success) {
        setCurrentStatus('Transcription complete!');
        onTranscriptionComplete(response.data);
      } else {
        onError(response.data.error || 'Transcription failed');
      }
    } catch (error) {
      console.error('Error:', error);
      if (error.response) {
        onError(error.response.data.error || 'Server error occurred');
      } else if (error.request) {
        onError('Cannot connect to server. Please ensure the backend is running on port 5000.');
      } else {
        onError('An unexpected error occurred');
      }
    } finally {
      setUploadProgress(0);
      setCurrentStatus('');
    }
  }, [onTranscriptionComplete, onTranscriptionStart, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    },
    multiple: false,
  });

  return (
    <div className="w-full max-w-4xl mx-auto">
      <motion.div
        {...getRootProps()}
        className={`glass-strong rounded-3xl p-12 md:p-16 cursor-pointer transition-all duration-300 ${
          isDragActive ? 'scale-105 glow' : 'hover:scale-102 glow-hover'
        }`}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input {...getInputProps()} />
        
        <div className="text-center">
          {/* Upload icon */}
          <motion.div
            className="mb-6"
            animate={isDragActive ? { scale: [1, 1.1, 1] } : {}}
            transition={{ duration: 0.5, repeat: isDragActive ? Infinity : 0 }}
          >
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 mb-4">
              <span className="text-6xl">
                {isDragActive ? '🎯' : '📁'}
              </span>
            </div>
          </motion.div>

          {/* Text */}
          <h2 className="text-2xl md:text-3xl font-bold mb-3 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            {isDragActive ? 'Drop it here!' : 'Upload Audio or Video'}
          </h2>
          
          <p className="text-gray-300 text-lg mb-6">
            {isDragActive
              ? 'Release to upload your file'
              : 'Drag & drop your file here, or click to browse'}
          </p>

          {/* Supported formats */}
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {['MP3', 'WAV', 'MP4', 'AVI', 'MOV'].map((format) => (
              <span
                key={format}
                className="px-4 py-2 glass rounded-full text-sm font-medium text-purple-300"
              >
                {format}
              </span>
            ))}
          </div>

          {/* Upload button */}
          <motion.button
            className="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Choose File
          </motion.button>

          {/* File size info */}
          <p className="text-gray-400 text-sm mt-6">
            Maximum file size: 500MB
          </p>
        </div>
      </motion.div>

      {/* Processing indicator */}
      {currentStatus && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 glass-strong rounded-2xl p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <span className="text-lg font-medium text-purple-300">
              {currentStatus}
            </span>
            <span className="text-sm text-gray-400">
              {uploadProgress > 0 && uploadProgress < 100 ? `${uploadProgress}%` : ''}
            </span>
          </div>
          
          {/* Progress bar */}
          <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: uploadProgress > 0 ? `${uploadProgress}%` : '100%' }}
              transition={{ duration: 0.5 }}
            />
          </div>

          {/* Processing animation */}
          {uploadProgress === 100 && (
            <div className="flex items-center justify-center mt-4 gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-2 h-2 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
