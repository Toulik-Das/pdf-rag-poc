import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response, initialize_pinecone_vectorstore
from dotenv import load_dotenv
import time
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
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT or Gemini Flash 1.5(Free Tier).")

# Sidebar for API Key, Model Selection, and PDF Upload
with st.sidebar:
    model_options = ["gpt-4o-mini", "gpt-4", "Gemini Flash 1.5(Free Tier)"]
    selected_model = st.selectbox("Select a model:", model_options)

    if selected_model == "Gemini Flash 1.5(Free Tier)":
        gemini_api_key = st.secrets["api_keys"]["gemini_key"]
        api_key = st.text_input("Enter your OpenAI API Key (To Generate Embeddings):", type="password")
    else:
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    use_pinecone = st.checkbox("Enable Collibra Knowledge")

# Function to send chat input to Gemini Flash 1.5 (Free Tier)
def get_gemini_response(user_input: str):
    try:
        genai.configure(api_key=gemini_api_key)
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(
            history=[{"role": "user", "parts": [user_input]}]
        )
        response = chat_session.send_message(user_input)
        return response.text
    except Exception as e:
        st.error(f"Error while fetching the Gemini response: {e}")
        return "There was an error processing your request with Gemini Flash 1.5."

# Main logic
if api_key:
    try:
        # Initialize FAISS vectorstore if files are uploaded
        if uploaded_files:
            st.write("Processing documents 🧾")
            documents = process_pdfs(uploaded_files)
            if documents:
                vectorstore_faiss = initialize_vectorstore(api_key, documents)
                st.write(f"Uploaded and processed {len(documents)} documents into the FAISS knowledge base.")
            else:
                st.warning("No valid documents were found in the uploaded files.")
        
        # Initialize Pinecone vectorstore only if the checkbox is checked
        vectorstore_pinecone = initialize_pinecone_vectorstore(PINECONE_API_KEY) if use_pinecone else None

        # Setup final vectorstore based on user input
        if vectorstore_faiss and vectorstore_pinecone:
            def retrieve_combined_knowledge(query):
                faiss_results = vectorstore_faiss.similarity_search(query)
                pinecone_results = vectorstore_pinecone.query(query, top_k=5, include_metadata=True)
                combined_results = faiss_results + pinecone_results["matches"]
                return combined_results
            
            retrieve_knowledge = retrieve_combined_knowledge
            st.write("Local & Specialised knowledge available for querying.")
        
        elif vectorstore_faiss:
            retrieve_knowledge = lambda query: vectorstore_faiss.similarity_search(query)
            st.write("Local knowledge available for querying.")
        
        elif vectorstore_pinecone:
            retrieve_knowledge = lambda query: vectorstore_pinecone.query(query, top_k=5, include_metadata=True)
            st.write("Connected For Specialised Knowledge Retrieval.")
        
        else:
            retrieve_knowledge = None
            st.warning("Please upload a PDF file or enable specialized knowledge to chat with the model.")
        
        # Chat history and input handling
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask a question about the content in your PDFs"):
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response_text = ""

                try:
                    if selected_model == "Gemini Flash 1.5(Free Tier)":
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

                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

        # Sidebar for chat history visibility
        with st.sidebar:
            if "show_chat_history" not in st.session_state:
                st.session_state["show_chat_history"] = False

            if st.button("Show/Hide Chat History"):
                st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

            if st.session_state["show_chat_history"]:
                st.write("### Chat History")
                for i, message in enumerate(st.session_state["chat_history"]):
                    st.write(f"**Q{i+1}:** {message['content']}" if message["role"] == "user" else f"**A{i+1}:** {message['content']}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your OpenAI API key to use the application.")
