import React from 'react';
import FloatingChatbot from '../components/FloatingChatbot';

// This component wraps the entire app and adds the floating chatbot
export default function Root({ children }) {
  return (
    <>
      {children}
      <FloatingChatbot />
    </>
  );
}
