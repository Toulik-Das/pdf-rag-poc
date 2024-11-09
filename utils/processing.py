import os 
import tempfile
import asyncio
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List, Generator

# Function to initialize vector store with FAISS only if documents are present
def initialize_vectorstore(api_key: str, documents: List) -> FAISS:
    
    db_name = "pdf_knowledge_base"
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    if os.path.exists(f"{db_name}.faiss"):
        # Load the existing FAISS index
        vectorstore = FAISS.load_local(db_name, embeddings)
    else:
        if documents:
            # Create the FAISS index only if there are documents to embed
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(db_name)
        else:
            raise ValueError("No documents found to initialize the FAISS vector store.")
    
    return vectorstore

# Function to load PDFs and split them into chunks
def process_pdfs(uploaded_files) -> List:
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getbuffer())
            loader = PyMuPDFLoader(tmp_file.name)
            docs = loader.load()
            docs = text_splitter.split_documents(docs)
            documents.extend(docs)
    
    return documents


# Function to get chat response with real-time streaming
def get_chat_response(user_input: str, vectorstore, model_name: str, api_key: str):
    # Initialize the ChatOpenAI model with streaming enabled
    llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0.7, stream=True)
    
    # Set up memory for the conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the ConversationalRetrievalChain
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    
    # Use the chain to generate a response in real-time
    response_stream = conversation_chain.stream({"question": user_input})

    # Handle each chunk based on its structure
    for chunk in response_stream:
        # Ensure chunk is not empty and handle as needed
        if chunk:
            # Check if chunk is a dictionary and contains 'content' as expected
            if isinstance(chunk, dict) and 'content' in chunk:
                yield chunk['content']
            # If chunk is not in the expected format, yield it directly as a fallback
            else:
                yield str(chunk)
