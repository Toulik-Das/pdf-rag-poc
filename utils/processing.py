import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Generator
from chromadb.config import Settings


# Function to initialize vector store
# def initialize_vectorstore(api_key: str):
#     db_name = "pdf_knowledge_base"
#     embeddings = OpenAIEmbeddings(api_key=api_key)
    
#     if os.path.exists(db_name):
#         vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
#     else:
#         vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
        
#     return vectorstore
def initialize_vectorstore(api_key: str):
    db_name = "pdf_knowledge_base"
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Initialize with settings if required
    settings = Settings(persist_directory=db_name)
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings, client_settings=settings)
    
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

# Function to get chat response (streaming)
def get_chat_response(user_input: str, vectorstore, model_name: str) -> Generator[str, None, None]:
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name, temperature=0.7, stream=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Yield responses as a generator
    for response in conversation_chain.stream({"question": user_input}):
        yield response["answer"]
