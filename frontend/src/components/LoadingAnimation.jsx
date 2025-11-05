import React from 'react';
import { motion } from 'framer-motion';

const LoadingAnimation = () => {
  return (
    <div className="w-full max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-strong rounded-3xl p-12 text-center"
      >
        {/* Animated microphone icon */}
        <motion.div
          className="mb-8 inline-block"
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 5, -5, 0],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <div className="w-32 h-32 mx-auto rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center pulse-glow">
            <span className="text-7xl">🎙️</span>
          </div>
        </motion.div>

        {/* Processing text */}
        <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
          Processing Your File
        </h2>

        {/* Status steps */}
        <div className="space-y-4 mb-8">
          <ProcessingStep 
            icon="📤" 
            text="Uploading file" 
            delay={0} 
          />
          <ProcessingStep 
            icon="🎵" 
            text="Extracting audio" 
            delay={0.2} 
          />
          <ProcessingStep 
            icon="🔇" 
            text="Reducing noise" 
            delay={0.4} 
          />
          <ProcessingStep 
            icon="🤖" 
            text="Transcribing with AI" 
            delay={0.6} 
          />
        </div>

        {/* Loading dots */}
        <div className="flex items-center justify-center gap-3">
          <motion.div
            className="w-4 h-4 bg-purple-500 rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0 }}
          />
          <motion.div
            className="w-4 h-4 bg-blue-500 rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
          />
          <motion.div
            className="w-4 h-4 bg-pink-500 rounded-full"
            animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
            transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
          />
        </div>

        <p className="text-gray-400 mt-8">
          This may take a few moments depending on file size...
        </p>
      </motion.div>
    </div>
  );
};

const ProcessingStep = ({ icon, text, delay }) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="flex items-center justify-center gap-3 text-gray-300"
    >
      <motion.span 
        className="text-2xl"
        animate={{ rotate: [0, 10, -10, 0] }}
        transition={{ duration: 2, repeat: Infinity, delay }}
      >
        {icon}
      </motion.span>
      <span className="text-lg">{text}</span>
      <motion.div
        className="ml-2"
        animate={{ opacity: [0, 1, 0] }}
        transition={{ duration: 1.5, repeat: Infinity, delay }}
      >
        <div className="flex gap-1">
          <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
          <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
          <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default LoadingAnimation;
