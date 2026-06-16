# 🎥 YouTube Transcript RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that enables users to converse with YouTube videos and retrieve timestamp-grounded insights across multiple videos.

The application combines transcript extraction, semantic search, conversational memory, and LLM-powered reasoning to provide explainable answers linked directly to relevant video segments.

---

## 🚀 Key Features

### 📺 Single Video Conversational Assistant

* Chat with any YouTube video using its transcript
* Multi-turn conversations with chat history awareness
* Follow-up question handling through query rewriting
* Transcript-grounded responses with optional LLM reasoning when context is insufficient

### 🔍 Multi-Video Knowledge Retrieval

* Search YouTube using a natural language query
* Automatically discover and retrieve relevant videos
* Aggregate information across multiple transcripts
* Return timestamp-grounded answers from different videos
* Surface direct video references for further exploration

### ⏱ Timestamp-Aware Retrieval

Unlike traditional transcript chunking approaches, this project preserves temporal information throughout the retrieval pipeline.

* Custom time-based transcript chunking
* Timestamp metadata attached to every chunk
* Source-grounded retrieval
* Direct navigation to relevant video segments

---

## 🏗️ System Architecture

### Single Video Mode

User Question
→ Query Rewriting
→ Transcript Retrieval
→ FAISS Similarity Search
→ LLM Response Generation
→ Conversational Memory Update

### Multi Video Mode

User Query
→ YouTube Video Discovery
→ Transcript Extraction
→ Timestamp-Aware Chunking
→ Embedding Generation
→ FAISS Vector Search
→ Multi-Video Retrieval
→ LLM Summarization
→ Timestamp-Grounded Answers

---

## 🛠️ Tech Stack

### LLM & Retrieval

* HuggingFace Inference API
* Meta Llama 3 8B Instruct
* Google Gemini 2.5 Flash
* LangChain
* FAISS Vector Store
* Sentence Transformers

### Data Sources

* YouTube Transcript API
* yt-dlp

### Frontend

* Streamlit

### Backend

* Python

---

## ⚙️ Setup

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Environment

#### Windows

```bash
myenv\Scripts\activate
```

#### Linux / MacOS

```bash
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=YOUR_TOKEN
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 💡 Engineering Challenges Solved

### Preserving Video Timestamps During Retrieval

Traditional text chunking loses temporal information, making it difficult to trace answers back to the source video.

A custom timestamp-aware chunker was implemented to:

* preserve transcript timing information
* maintain source attribution
* enable timestamp-grounded responses

### Multi-Video Answer Aggregation

Responses are consolidated across multiple videos while limiting duplication and prioritizing the most relevant evidence from each source.

### Conversational Retrieval

Follow-up questions are rewritten into standalone queries before retrieval, improving retrieval quality and enabling natural conversations over video content.

---

## 🔮 Future Improvements

* Hybrid Search (Dense + BM25)
* Cross-Encoder Reranking
* Persistent Vector Database (Chroma / Qdrant)
* Streaming Responses
* Whisper-based Audio Transcription
* Multi-language Support
* Agentic Video Research Workflow
* Evaluation Pipeline for Retrieval Quality

---

## 🏷️ Topics

`rag` `genai` `langchain` `faiss` `huggingface` `streamlit` `youtube` `chatbot` `retrieval-augmented-generation` `semantic-search` `vector-database` `llm` `conversational-ai`
