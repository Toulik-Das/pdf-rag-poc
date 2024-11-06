import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
import os
from dotenv import load_dotenv
import sys

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
        response_placeholder = st.empty()  # Placeholder for streaming response
        response_text = ""  # To accumulate streamed responses
        
        # Stream the response
        for chunk in get_chat_response(user_input, vectorstore, selected_model, api_key):
            response_text += chunk  # Accumulate streamed text
            response_placeholder.write(response_text)  # Update the UI with the new chunk

        st.session_state["chat_history"].append((user_input, response_text))
    
    # Display chat history
    if st.session_state["chat_history"]:
        st.write("### Chat History")
        for i, (question, answer) in enumerate(st.session_state["chat_history"]):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {answer}")
else:
    st.warning("Please enter your OpenAI API key to use the application.")
