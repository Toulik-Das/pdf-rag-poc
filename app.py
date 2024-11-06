import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
from dotenv import load_dotenv

# Load environment variables (if needed for other settings)
load_dotenv()

st.set_page_config(
    page_title="PDF-based RAG Knowledge Worker",
    page_icon="📚",
    layout="wide",
)

# Set up Streamlit UI
st.title("PDF-based RAG Knowledge Worker")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT.")

# Sidebar for API Key, PDF upload, and Model Selection
with st.sidebar:
    # Input for OpenAI API Key
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Model selection for OpenAI
    model_options = ["gpt-4o-mini", "gpt-4"]
    selected_model = st.selectbox("Select a model:", model_options)
    
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Initialize vectorstore and process PDFs only if the API key is provided
if api_key:
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

    # Initialize a flag for toggling chat history visibility
    if "show_chat_history" not in st.session_state:
        st.session_state["show_chat_history"] = False

    # Style the chat window using custom CSS for dark mode
    st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .chat-box {
            width: 100%;
            max-width: 700px;
            height: 400px;
            border: 1px solid #444;
            border-radius: 10px;
            display: flex;
            flex-direction: column;
            padding: 10px;
            overflow-y: scroll;
            background-color: #333;
        }
        .chat-box .message {
            margin: 5px 0;
            padding: 8px;
            border-radius: 10px;
            max-width: 80%;
        }
        .user-message {
            background-color: #4a90e2;
            align-self: flex-end;
            color: white;
        }
        .gpt-message {
            background-color: #555;
            align-self: flex-start;
            color: white;
        }
        .input-container {
            width: 100%;
            margin-top: 10px;
        }
        .stTextInput input {
            background-color: #2c2c2c;
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #357ab7;
        }
    </style>
    """, unsafe_allow_html=True)

    # User Input
    user_input = st.text_input("Ask a question about the content in your PDFs:")

    # Display chat in a "chat box" style
    with st.container():
        # Display all chat history
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)

        # Display previous messages (question and answer)
        for i, (question, answer) in enumerate(st.session_state["chat_history"]):
            st.markdown(f'<div class="message user-message">{question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="message gpt-message">{answer}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Submit question and get model answer
    if user_input:
        # Get chat response (no streaming, just full response)
        response_text = get_chat_response(user_input, vectorstore, selected_model, api_key)

        # Display the new response and add to chat history
        st.session_state["chat_history"].append((user_input, response_text))

    # Sidebar to show chat history
    with st.sidebar:
        # Toggle chat history visibility
        if st.button("Show/Hide Chat History"):
            st.session_state["show_chat_history"] = not st.session_state["show_chat_history"]

        if st.session_state["show_chat_history"]:
            st.write("### Chat History")
            for i, (question, answer) in enumerate(st.session_state["chat_history"]):
                st.write(f"**Q{i+1}:** {question}")
                st.write(f"**A{i+1}:** {answer}")
else:
    st.warning("Please enter your OpenAI API key to use the application.")
