import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response, initialize_pinecone_vectorstore
from dotenv import load_dotenv
import time
import openai
import google.generativeai as genai  # Gemini integration

# Load environment variables
load_dotenv()
PINECONE_API_KEY = st.secrets["api_keys"]["PINECONE_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="📚 QueryWise",
    page_icon="📘",
    layout="wide",
)

# Title and description
st.title("QueryWise 🧠")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT or Gemini Flash 1.5 (Free Tier).")

# Sidebar for API Key, Model Selection, and PDF Upload
with st.sidebar:
    # Model selection for OpenAI and Gemini
    model_options = ["gpt-4o-mini", "gpt-4", "Gemini Flash 1.5 (Free Tier)"]
    selected_model = st.selectbox("Select a model:", model_options)

    # Automatically use the API key from secrets if Gemini Flash 1.5 is selected
    if selected_model == "Gemini Flash 1.5 (Free Tier)":
        gemini_api_key = st.secrets["api_keys"]["gemini_key"]
        api_key = st.text_input("Enter your OpenAI API Key (To Generate Embeddings):", type="password")
    else:
        # Prompt user to input OpenAI API key for other models
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    use_pinecone = st.checkbox("Enable Collibra Knowledge")

# Function to get embeddings for queries
def get_embedding(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Adjust model as needed
    )
    return response['data'][0]['embedding']

# Initialize vectorstores and process PDFs if API key is provided
if api_key:
    try:
        # Process uploaded PDFs
        if uploaded_files:
            st.write("Processing documents 🧾 ")
            documents = process_pdfs(uploaded_files)

            if documents:
                # Initialize FAISS vectorstore with documents
                vectorstore_faiss = initialize_vectorstore(api_key, documents)
                st.write(f"Uploaded and processed {len(documents)} documents into the FAISS knowledge base.")
            else:
                st.warning("No valid documents were found in the uploaded files.")
        
            # Initialize Pinecone if enabled
            if use_pinecone:
                vectorstore_pinecone = initialize_pinecone_vectorstore(PINECONE_API_KEY)
                st.write("Connected for Specialized Knowledge Retrieval.")

        elif use_pinecone:
            # Initialize only Pinecone if no PDFs are uploaded
            vectorstore_faiss = None
            vectorstore_pinecone = initialize_pinecone_vectorstore(PINECONE_API_KEY)
            st.write("Connected for Specialized Knowledge Retrieval.")
        else:
            vectorstore_faiss = None
            vectorstore_pinecone = None
            st.warning("Please upload a PDF file or enable specialized knowledge to chat with the model.")

        # Define retrieve_knowledge function based on available vectorstores
        if vectorstore_faiss and vectorstore_pinecone:
            def retrieve_combined_knowledge(query):
                # Get vector for the query
                query_vector = get_embedding(query, api_key)
                
                # Search FAISS vectorstore
                faiss_results = vectorstore_faiss.similarity_search(query)
                
                # Search Pinecone vectorstore
                pinecone_response = vectorstore_pinecone.query(
                    namespace="ns1",
                    vector=query_vector,
                    top_k=5,
                    include_values=True,
                    include_metadata=True
                )
                pinecone_results = pinecone_response.get("matches", [])
                
                # Combine FAISS and Pinecone results
                combined_results = faiss_results + pinecone_results
                return combined_results
            
            retrieve_knowledge = retrieve_combined_knowledge
            st.write("Local & Specialized knowledge available for querying.")
        
        elif vectorstore_faiss:
            retrieve_knowledge = lambda query: vectorstore_faiss.similarity_search(query)
            st.write("Local knowledge available for querying.")
        
        elif vectorstore_pinecone:
            def retrieve_pinecone_only(query):
                query_vector = get_embedding(query, api_key)
                pinecone_response = vectorstore_pinecone.query(
                    namespace="ns1",
                    vector=query_vector,
                    top_k=5,
                    include_values=True,
                    include_metadata=True
                )
                return pinecone_response.get("matches", [])
            
            retrieve_knowledge = retrieve_pinecone_only
            st.write("Connected for Specialized Knowledge Retrieval.")

        else:
            retrieve_knowledge = None
            st.warning("Please upload a PDF file or enable specialized knowledge to chat with the model.")
        
        # Chat history management
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display chat messages from history
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if user_input := st.chat_input("Ask a question about the content in your PDFs"):
            # Add user message to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Display assistant response with simulated streaming
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_text = ""

                try:
                    if selected_model == "Gemini Flash 1.5 (Free Tier)":
                        for chunk in get_chat_response(user_input, retrieve_knowledge, selected_model, gemini_api_key):
                            response_text += chunk
                            response_placeholder.markdown(response_text)
                            time.sleep(0.05)
                    else:
                        for chunk in get_chat_response(user_input, retrieve_knowledge, selected_model, api_key):
                            response_text += chunk
                            response_placeholder.markdown(response_text)
                            time.sleep(0.05)

                except Exception as e:
                    st.error(f"Error while fetching the response: {e}")
                    response_placeholder.markdown("There was an error processing your request.")

                # Save the assistant's final response in markdown format for chat history
                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

        # Sidebar to toggle chat history visibility
        with st.sidebar:
            if "show_chat_history" not in st.session_state:
                st.session_state["show_chat_history"] = False

            if st.button("Show/Hide Chat History"):
                st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

            # Show chat history in sidebar if toggled on
            if st.session_state["show_chat_history"]:
                st.write("### Chat History")
                for i, message in enumerate(st.session_state["chat_history"]):
                    if message["role"] == "user":
                        st.write(f"**Q{i+1}:** {message['content']}")
                    else:
                        st.write(f"**A{i+1}:** {message['content']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please enter your OpenAI API key to use the application.")
