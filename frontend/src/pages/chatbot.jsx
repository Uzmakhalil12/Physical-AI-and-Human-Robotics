import React from 'react';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Chatbot from '@site/src/components/Chatbot';

export default function ChatbotPage() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Physical AI & Human Robotics Chatbot`}
      description="A RAG chatbot for Physical AI and Human Robotics">
      <main>
        <div style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
          <h1>Physical AI & Human Robotics Assistant</h1>
          <p>
            This chatbot is designed to answer questions specifically about Physical AI, Human-Robotics, and related topics.
            It uses a Retrieval-Augmented Generation (RAG) system to provide accurate responses based on a curated knowledge base.
          </p>
          <div style={{ marginTop: '2rem' }}>
            <Chatbot />
          </div>
        </div>
      </main>
    </Layout>
  );
}