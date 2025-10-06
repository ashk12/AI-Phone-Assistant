# 📱 AI Phone Assistant

A conversational AI assistant that helps users find, compare, and learn about mobile phones. Built with FastAPI backend and React frontend. Supports structured and semantic queries, with robust safety and prompt handling.

---

## 🚀 Features

- Phone recommendations under constraints (budget, camera, battery, OS, etc.)
- Phone comparisons and detailed specifications
- Explanations of phone features / technical concepts
- Safety checks to prevent malicious or ambiguous queries
- Semantic search for natural language queries
- Streaming responses (optional future enhancement)

---

## 🛠 Tech Stack

- **Backend:** Python, FastAPI, Scikit-learn (TF-IDF, cosine similarity), Gemini API
- **Frontend:** React, React Markdown
- **Data:** Local JSON / GitHub-hosted JSON file of products
- **Deployment:** Vercel (frontend & backend)

---

## 🏗 Architecture Overview

1. **Frontend** (React):
   - Chat interface similar to ChatGPT light theme
   - Handles user input and displays AI responses

2. **Backend** (FastAPI):
   - `MultiIntentPhoneSystem` handles query processing
   - Intent detection via Gemini API
   - Semantic & structured search on product data
   - Optional streaming response endpoint

3. **Data**:
   - Products JSON loaded from GitHub or local file
   - Product features include brand, camera, battery, RAM, OS, etc.

---

## 💡 Prompt Design & Safety Strategy

- Queries are classified for intent: `recommendation`, `comparison`, `details`, `explanation`
- Safety checks:
  - Local function `is_unsafe_query`
  - Gemini API classification for toxic/irrelevant content
- Prompts are structured to:
  - Return JSON for intent detection
  - Generate concise and structured recommendations/comparisons

---

## ⚙️ Setup Instructions

1. Clone repository:

```bash
git clone https://github.com/ashk12/AI-Phone-Assistant.git
cd AI-Phone-Assistant
