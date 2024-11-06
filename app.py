import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
from dotenv import load_dotenv
import os
import time
import streamlit_authenticator as stauth
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables from .env file
load_dotenv()


# Retrieve the Google client ID, client secret, and cookie name from environment variables
google_client_id = os.getenv("GOOGLE_CLIENT_ID")
google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
cookie_name = os.getenv("COOKIE_NAME")

# Initialize Firebase for chat history storage
firebase_credentials_path = "edm-knowledge-firebase-adminsdk-wxb5x-aed415c3d2.json"
cred = credentials.Certificate(firebase_credentials_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Page configuration
st.set_page_config(
    page_title="📚 QueryWise",
    page_icon="📘",
    layout="wide",
)

# Title and description
st.title("QueryWise 🧠")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT.")

# Google login setup (use streamlit-authenticator or any other method)
authenticator = stauth.Authenticate(
    google_client_id=google_client_id,
    google_client_secret=google_client_secret,
    cookie_name=cookie_name
)

# Sidebar for API Key, Model Selection, and PDF Upload
with st.sidebar:
    # Input for OpenAI API Key
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    
    # Model selection for OpenAI
    model_options = ["gpt-4o-mini", "gpt-4"]
    selected_model = st.selectbox("Select a model:", model_options)
    
    # Process PDF upload
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

# Check if the user is logged in
user_info = authenticator.login("Sign in with Google")

if user_info:
    username = user_info["username"]

    # Retrieve the user's previous chat history from Firestore
    chat_ref = db.collection("chat_history").document(username)
    chat_history_doc = chat_ref.get()
    if chat_history_doc.exists:
        st.session_state["chat_history"] = chat_history_doc.to_dict().get("messages", [])
    else:
        st.session_state["chat_history"] = []

    try:
        if api_key:
            if uploaded_files:
                st.write("Processing documents 🧾 ")
                documents = process_pdfs(uploaded_files)

                if documents:
                    # Initialize vectorstore with documents
                    vectorstore = initialize_vectorstore(api_key, documents)
                    st.write(f"Uploaded and processed {len(documents)} documents into the knowledge base.")
                else:
                    st.warning("No valid documents were found in the uploaded files.")
            
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
                        # Get and display the response in a streaming fashion, handling each new sentence or section as markdown
                        for chunk in get_chat_response(user_input, vectorstore, selected_model, api_key):
                            response_text += chunk
                            response_placeholder.markdown(response_text)  # Update full markdown output so far
                            time.sleep(0.05)  # Simulate streaming effect
                    except Exception as e:
                        st.error(f"Error while fetching the response: {e}")
                        response_placeholder.markdown("There was an error processing your request.")

                    # Save the assistant's final response in markdown format for chat history
                    st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

                # Save updated chat history to Firestore for persistent storage
                chat_ref.set({"messages": st.session_state["chat_history"]})

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
    st.warning("Please log in to access the application.")
