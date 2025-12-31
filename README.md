# Physical AI and Human Robotics Book

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator. It contains comprehensive materials for Physical AI and Human Robotics education.

## Features

- Complete Physical AI & Human Robotics course materials
- Interactive documentation with Docusaurus
- **AI-powered RAG Chatbot** - Ask questions about Physical AI & Human Robotics and get accurate answers from our knowledge base

## Chatbot Integration

A Retrieval-Augmented Generation (RAG) chatbot has been integrated into this site. Access it at the "Chat with AI Assistant" link in the navigation sidebar. The chatbot is specifically trained on Physical AI & Human Robotics content and will respond to out-of-scope questions appropriately.

For more details on the chatbot implementation, see [README-RAG-CHATBOT.md](README-RAG-CHATBOT.md).

## Installation

```bash
cd frontend
yarn
```

## Local Development

For the frontend (book site):
```bash
cd frontend
yarn start
```

For the backend (RAG chatbot API):
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

For the frontend (book site):
```bash
cd frontend
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
cd frontend
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
cd frontend
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
