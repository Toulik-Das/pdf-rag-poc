import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="📚 QueryWise",
    page_icon="📘",
    layout="wide",
)

# def stream_response_smoothly(full_response: str):
#     """Function to smoothly stream the response in Streamlit by writing character by character."""
#     response_placeholder = st.empty()
#     response_text = ""

#     # Iterate through each chunk of the response and stream it
#     for chunk in full_response:
#         response_text += chunk
#         response_placeholder.markdown(response_text)  # Update the response incrementally
#         time.sleep(0.01)  # Adjust the delay for smoother or faster typing effect

def stream_response_smoothly(full_response: str):
    """Function to smoothly stream the response in Streamlit by writing text line by line."""
    response_placeholder = st.empty()
    response_text = ""
    
    # Iterate through each line in the response
    for char in full_response:
        response_text += char
        response_placeholder.markdown(response_text)  # Update the response incrementally
        time.sleep(0.01)  # Adjust the delay for smoother or faster typing effect
        
# Title and description
st.title("QueryWise 🧠")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT.")

# Sidebar for API Key, Model Selection, and PDF Upload
with st.sidebar:
    # Model selection for OpenAI
    model_options = ["gpt-4o-mini", "gpt-4", "Gemini Flash 1.5(Free Tier)"]
    selected_model = st.selectbox("Select a model:", model_options)

    # Automatically use the API key from secrets if Gemini Flash 1.5 is selected
    if selected_model == "gemini-1.5-flash":
        api_key = st.secrets["api_keys"]["gemini_key"]
    else:
        # Prompt user to input OpenAI API key for other models
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Initialize vectorstore and process PDFs only if the API key is provided
if api_key:
    try:
        if uploaded_files:
            st.write("Processing documents 🧾 ")
            documents = process_pdfs(uploaded_files)

            if documents:
                # Initialize vectorstore with documents
                vectorstore = initialize_vectorstore(api_key, documents)
                st.write(f"Uploaded and processed {len(documents)} documents into the knowledge base.")
            else:
                st.warning("No valid documents were found in the uploaded files.")
        
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
        
            # Get and stream the assistant response smoothly
            with st.chat_message("assistant"):
                try:
                    # Get the full response in chunks
                    full_response_chunks = get_chat_response(user_input, vectorstore, selected_model, api_key)
        
                    # Stream the response smoothly
                    for chunk in full_response_chunks:
                        stream_response_smoothly(chunk)
        
                    # Save the assistant's final response in chat history
                    st.session_state["chat_history"].append({"role": "assistant", "content": chunk})
        
                except Exception as e:
                    st.error(f"Error while fetching the response: {e}")

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
