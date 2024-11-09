import os
import asyncio
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List

# Function to initialize vector store with FAISS only if documents are present
def initialize_vectorstore(api_key: str, documents: List) -> FAISS:
    db_name = "pdf_knowledge_base"
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Check if the vectorstore already exists, if so, load it
    if os.path.exists(f"{db_name}.faiss"):
        vectorstore = FAISS.load_local(db_name, embeddings)
    else:
        if documents:
            # Create the FAISS index only if documents are provided
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
        loader = PyMuPDFLoader(file)
        docs = loader.load()
        docs = text_splitter.split_documents(docs)
        documents.extend(docs)
    
    return documents

def get_chat_response(user_input: str, vectorstore, model_name: str, api_key: str):
    llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0.7)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    response = conversation_chain({"question": user_input})

    # Check if the response has an 'answer' field
    if 'answer' in response:
        return response['answer']
    else:
        raise ValueError(f"Unexpected response format: {response}")
