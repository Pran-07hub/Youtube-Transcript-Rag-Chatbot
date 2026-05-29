# 🎥 YouTube Transcript RAG Chatbot

A GenAI-powered YouTube chatbot that:

* fetches YouTube transcripts
* performs semantic retrieval
* supports multi-video querying
* provides timestamp-grounded answers
* uses HuggingFace LLMs + FAISS + LangChain

---

# Features

## 🔍 Multi Video Search

* User enters a query
* System searches YouTube videos
* Retrieves transcripts
* Performs chunking + embeddings
* Returns timestamp-aware summarized answers

## 📺 Single Video QA

* User provides a YouTube video URL
* Ask questions specifically from that video in the form of a chat

---

# Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace
* YouTube Transcript API
* yt-dlp

---

# Setup

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

## Create Virtual Environment

```bash
python -m venv myenv
```

## Activate Environment

### Windows

```bash
myenv\Scripts\activate
```

### Mac/Linux

```bash
source myenv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App

```bash
streamlit run app.py
```

---

# Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=YOUR_TOKEN
```

---

# Future Improvements

* Hybrid Retrieval
* Reranking
* Chat History
* Streaming Responses
* Persistent Vector DB
* Whisper Integration
* Async Processing