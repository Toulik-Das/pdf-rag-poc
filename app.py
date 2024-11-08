import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response, initialize_pinecone_vectorstore
from dotenv import load_dotenv
import time
import google.generativeai as genai  # Gemini integration
import pinecone

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
    # Model selection for OpenAI and Gemini
    model_options = ["gpt-4o-mini", "gpt-4", "Gemini Flash 1.5(Free Tier)"]
    selected_model = st.selectbox("Select a model:", model_options)

    # Automatically use the API key from secrets if Gemini Flash 1.5 is selected
    if selected_model == "Gemini Flash 1.5(Free Tier)":
        gemini_api_key = st.secrets["api_keys"]["gemini_key"]
        api_key = st.text_input("Enter your OpenAI API Key(To Generate Embeddings) :", type="password")
    else:
        # Prompt user to input OpenAI API key for other models
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    use_pinecone = st.checkbox("Enable Collibra Knowledge")

# Function to send chat input to Gemini Flash 1.5 (Free Tier)
def get_gemini_response(user_input: str):
    try:
        # Configure Gemini Flash 1.5 API key
        genai.configure(api_key=gemini_api_key)

        # Create a chat session for Gemini
        chat_session = genai.GenerativeModel(model_name="gemini-1.5-flash").start_chat(
            history=[{"role": "user", "parts": [user_input]}]
        )

        # Send message and receive response
        response = chat_session.send_message(user_input)

        return response.text
    except Exception as e:
        st.error(f"Error while fetching the Gemini response: {e}")
        return "There was an error processing your request with Gemini Flash 1.5."

# Initialize vectorstore and process PDFs only if the API key is provided
if api_key:
    try:
        vectorstore = None
        
        # Case where PDF files are uploaded
        if uploaded_files:
            st.write("Processing documents 🧾 ")
            documents = process_pdfs(uploaded_files)

            if documents:
                # Initialize FAISS vectorstore with documents
                vectorstore_faiss = initialize_vectorstore(api_key, documents)
                st.write(f"Uploaded and processed {len(documents)} documents into the FAISS knowledge base.")                
            else:
                st.warning("No valid documents were found in the uploaded files.")

            if use_pinecone:
                # Initialize Pinecone vectorstore for knowledge retrieval (no documents added here)
                vectorstore_pinecone = initialize_pinecone_vectorstore(PINECONE_API_KEY)
                st.write("Connected For Specialised Knowledge Retrieval")
                
                # You will use the Pinecone query method instead of 'as_retriever'
                def query_pinecone(query):
                    # Example: Replace this with actual query functionality based on your setup
                    response = vectorstore_pinecone.query(
                        query_vector=query, 
                        top_k=5,  # Number of similar documents to retrieve
                        include_metadata=True
                    )
                    return response

                # Combine FAISS retrieval and Pinecone query manually
                def combined_retrieval(query):
                    # First use FAISS retriever
                    faiss_results = vectorstore_faiss.as_retriever().retrieve(query)
                    
                    # Then use Pinecone query
                    pinecone_results = query_pinecone(query)
                    
                    # Combine or rank results from FAISS and Pinecone
                    return faiss_results + pinecone_results
                
                vectorstore = combined_retrieval
                st.write("Local & Specialised knowledge available for querying.")
        
        # Case where no PDFs are uploaded, but Pinecone is enabled
        if use_pinecone:
            # Initialize Pinecone vectorstore for knowledge retrieval
            vectorstore_pinecone = initialize_pinecone_vectorstore(PINECONE_API_KEY)
            if vectorstore is None:  # If vectorstore wasn't initialized by PDFs, initialize with Pinecone
                vectorstore = query_pinecone  # Use the Pinecone query method directly
            st.write("Connected For Specialised Knowledge Retrieval.")
        
        # Check if vectorstore is initialized before proceeding
        if vectorstore is None:
            st.warning("Vectorstore not initialized properly. Please upload PDFs or enable Pinecone for querying.")
            st.stop()  # Stop further execution if vectorstore isn't initialized

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
                    if selected_model == "Gemini Flash 1.5(Free Tier)":
                        # Get and display the response from Gemini Flash 1.5
                        for chunk in get_chat_response(user_input, vectorstore, selected_model, gemini_api_key):
                            response_text += chunk
                            response_placeholder.markdown(response_text)  # Update full markdown output so far
                            time.sleep(0.05)  # Simulate streaming effect
                    else:
                        # Get and display the response for GPT-based models
                        for chunk in get_chat_response(user_input, vectorstore, selected_model, api_key):
                            response_text += chunk
                            response_placeholder.markdown(response_text)  # Update full markdown output so far
                            time.sleep(0.05)  # Simulate streaming effect

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
