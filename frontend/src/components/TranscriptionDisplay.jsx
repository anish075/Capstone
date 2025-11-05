import React, { useState } from 'react';
import { motion } from 'framer-motion';

const TranscriptionDisplay = ({ transcription, onReset }) => {
  const [copied, setCopied] = useState(false);

  const formatTimestamp = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(transcription.transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const element = document.createElement('a');
    const file = new Blob([transcription.transcription], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `transcription_${transcription.filename || 'file'}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleDownloadWithTimestamps = () => {
    let content = `Transcription for: ${transcription.filename}\n`;
    content += `Language: ${transcription.language}\n`;
    content += `\n${'='.repeat(50)}\n\n`;
    
    transcription.segments.forEach((segment) => {
      content += `[${formatTimestamp(segment.start)} - ${formatTimestamp(segment.end)}]\n`;
      content += `${segment.text}\n\n`;
    });

    const element = document.createElement('a');
    const file = new Blob([content], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `transcription_with_timestamps_${transcription.filename || 'file'}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-strong rounded-2xl p-6 mb-6"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <span className="text-4xl">✅</span>
            <div>
              <h2 className="text-2xl font-bold text-green-400">Transcription Complete</h2>
              <p className="text-gray-400 text-sm mt-1">
                File: {transcription.filename} • Language: {transcription.language.toUpperCase()}
              </p>
            </div>
          </div>

          <button
            onClick={onReset}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold hover:from-purple-500 hover:to-blue-500 transition-all duration-300 whitespace-nowrap"
          >
            🔄 New Upload
          </button>
        </div>
      </motion.div>

      {/* Action buttons */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex flex-wrap gap-3 mb-6"
      >
        <button
          onClick={handleCopy}
          className="flex-1 min-w-[150px] px-6 py-3 glass-strong rounded-xl font-semibold hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
        >
          <span className="text-xl">{copied ? '✅' : '📋'}</span>
          {copied ? 'Copied!' : 'Copy Text'}
        </button>

        <button
          onClick={handleDownload}
          className="flex-1 min-w-[150px] px-6 py-3 glass-strong rounded-xl font-semibold hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
        >
          <span className="text-xl">💾</span>
          Download
        </button>

        <button
          onClick={handleDownloadWithTimestamps}
          className="flex-1 min-w-[150px] px-6 py-3 glass-strong rounded-xl font-semibold hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2"
        >
          <span className="text-xl">⏱️</span>
          With Timestamps
        </button>
      </motion.div>

      {/* Full transcription */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-strong rounded-2xl p-8 mb-6"
      >
        <h3 className="text-xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
          📝 Full Transcription
        </h3>
        <div className="prose prose-invert max-w-none">
          <p className="text-gray-200 text-lg leading-relaxed whitespace-pre-wrap">
            {transcription.transcription}
          </p>
        </div>
      </motion.div>

      {/* Timestamped segments */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-strong rounded-2xl p-8"
      >
        <h3 className="text-xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
          ⏱️ Timestamped Segments
        </h3>
        
        <div className="space-y-4 max-h-[600px] overflow-y-auto pr-4">
          {transcription.segments.map((segment, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 + index * 0.05 }}
              className="glass rounded-xl p-4 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <div className="px-3 py-1 bg-purple-600/30 rounded-lg text-sm font-mono">
                    {formatTimestamp(segment.start)}
                  </div>
                  <div className="text-center text-xs text-gray-500 mt-1">to</div>
                  <div className="px-3 py-1 bg-blue-600/30 rounded-lg text-sm font-mono">
                    {formatTimestamp(segment.end)}
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-gray-200 leading-relaxed">
                    {segment.text}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        <div className="glass-strong rounded-xl p-6 text-center">
          <div className="text-3xl font-bold text-purple-400">
            {transcription.segments.length}
          </div>
          <div className="text-gray-400 text-sm mt-2">Segments</div>
        </div>
        
        <div className="glass-strong rounded-xl p-6 text-center">
          <div className="text-3xl font-bold text-blue-400">
            {transcription.transcription.split(' ').length}
          </div>
          <div className="text-gray-400 text-sm mt-2">Words</div>
        </div>
        
        <div className="glass-strong rounded-xl p-6 text-center">
          <div className="text-3xl font-bold text-pink-400">
            {formatTimestamp(transcription.segments[transcription.segments.length - 1]?.end || 0)}
          </div>
          <div className="text-gray-400 text-sm mt-2">Duration</div>
        </div>
      </motion.div>
    </div>
  );
};

export default TranscriptionDisplay;
