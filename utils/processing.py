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
from pinecone import Pinecone

# Function to initialize Pinecone vectorstore for knowledge retrieval
def initialize_pinecone_vectorstore(PINECONE_API_KEY: str):
    pc = Pinecone(api_key=PINECONE_API_KEY, environment="production")
    index = pc.Index("rag-poc")
    return index

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

# Function to get chat response without streaming
def get_chat_response(user_input: str, vectorstore, pinecone_index, model_name: str, api_key: str):
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0.7)

    # Memory for the conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Get the retriever from the vectorstore
    #retriever = vectorstore.as_retriever()

    # NEW: Retrieve from FAISS
    faiss_retriever = vectorstore.as_retriever()
    faiss_results = faiss_retriever.get_relevant_documents(user_input)

    # NEW: Retrieve from Pinecone
    query_vector = OpenAIEmbeddings(api_key=api_key).embed_documents([user_input])[0]
    pinecone_results = pinecone_index.query(
        vector=query_vector,
        top_k=5,  # Top 5 results
        include_values=True,
        include_metadata=True
    )
    
    # Process Pinecone results to extract text from metadata if available, else from values
    pinecone_documents = []
    for match in pinecone_results['matches']:
        text_content = match.get('metadata', {}).get('text', '')  # Use metadata if available
        if not text_content:  # If metadata text is not available, fallback to `values`
            text_content = " ".join(map(str, match.get('values', [])))
        pinecone_documents.append({"text": text_content})
        
    combined_results = faiss_results + [Document(page_content=doc['text']) for doc in pinecone_documents]

    # Create the conversation chain
    #conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # NEW: Create conversation chain with combined results
    retriever = FAISS.from_documents(combined_results, OpenAIEmbeddings(api_key=api_key)).as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # Get the full response (in a streaming fashion)
    response = conversation_chain({"question": user_input})

    # Simulate yielding portions of the response as markdown-compatible chunks
    for sentence in response['answer'].split('. '):  # Adjust this split as needed to control chunk size
        yield sentence + '. '  # Yield each sentence followed by a period and space for clarity
