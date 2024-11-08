# 📚 QueryWise

**QueryWise** is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDFs, ask questions, and get AI-powered responses tailored to the content in the documents. Built with GPT-powered question-answering capabilities, QueryWise helps you turn static documents into interactive knowledge bases.

## 🚀 Features

- **Document Upload**: Securely upload PDFs for content analysis.
- **Dynamic Q&A**: Ask questions, and receive answers based on your PDFs’ content.
- **Streamed Responses**: Real-time, conversational responses powered by GPT.
- **Chat History**: Review past interactions and easily toggle chat visibility.
- **Data Security**: Only users with an API key can interact with the system.

## 🎨 Architecture Overview

### 1. **User Interface**  
   **Streamlit** is used for a clean, interactive user experience:
   - **PDF Upload**: Allows users to upload documents for analysis.
   - **Q&A Input**: Text input where users can ask questions.
   - **Chat Display**: Shows past questions and answers in a familiar chat format.

### 2. **Backend Processing**  
   - **PDF Processing**: Extracts text and metadata from PDFs, preparing them for question-answering.
   - **Vector Store Initialization**: Stores processed document embeddings in a vector database.
   - **OpenAI GPT Model**: Used to generate answers to user queries by pulling context from relevant document embeddings.

### 3. **Data Storage**  
   - **Vector Database**: Stores document embeddings for efficient retrieval.
   - **Session State Management**: Maintains chat history and user settings in a temporary session.

### 4. **External API Integration**  
   - **OpenAI API**: Provides GPT-based language model capabilities to answer user questions based on document content.

---

### 🛠️ Tech Stack

- **Streamlit** for UI
- **Python** for backend
- **OpenAI API** for language model
- **Vector Database** (Chroma or FAISS) for document embedding storage
- **dotenv** for environment variable management

## 📂 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Toulik-Das/pdf-rag-poc.git
   cd QueryWise
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**:  Create a ```.env``` file and add your OpenAI API key.
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```
4. **Run the app**:
   ```bash
   streamlit run app.py
   ```
---

### 🌐 Usage
  - Upload PDF: Drag and drop PDFs for processing.
  - Ask a Question: Enter a question about the PDF content.
  - View Response: See answers streamed in a chat format.

---

### 🤝 Contributing
  - Fork the repository.
  - Create a new branch (feature/your-feature).
  - Push changes and submit a Pull Request.
