import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
import os
from dotenv import load_dotenv


# Load environment variables (if needed for other settings)
load_dotenv()

# Set up Streamlit UI
st.title("PDF-based RAG Knowledge Worker")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT.")

# Input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Model selection for OpenAI
model_options = ["gpt-4o-mini", "gpt-4"]
selected_model = st.selectbox("Select a model:", model_options)

# Initialize vectorstore and process PDFs only if the API key is provided
if api_key:
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write("Processing documents...")
        documents = process_pdfs(uploaded_files)

        if documents:
            # Initialize vectorstore with documents
            vectorstore = initialize_vectorstore(api_key, documents)
            st.write(f"Uploaded and processed {len(documents)} documents into the knowledge base.")
        else:
            st.warning("No valid documents were found in the uploaded files.")
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask a question about the content in your PDFs:")
    
    if user_input:
        # Get chat response (no streaming, just full response)
        response_text = get_chat_response(user_input, vectorstore, selected_model, api_key)
        
        # Display response and save to chat history
        st.write("### Response")
        st.write(response_text)
        
        # Save question and response to session state
        st.session_state["chat_history"].append((user_input, response_text))
    
    # Sidebar to show chat history
    with st.sidebar:
        # Button to toggle chat history visibility
        if st.button("Show Chat History"):
            st.write("### Chat History")
            for i, (question, answer) in enumerate(st.session_state["chat_history"]):
                st.write(f"**Q{i+1}:** {question}")
                st.write(f"**A{i+1}:** {answer}")
else:
    st.warning("Please enter your OpenAI API key to use the application.")
