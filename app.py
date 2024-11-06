import streamlit as st
from utils.processing import process_pdfs, initialize_vectorstore, get_chat_response
import os
from dotenv import load_dotenv
import openai

# Load environment variables (if needed for other settings)
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="📚 PDF Expert", page_icon="📘", layout="wide")
st.title("📚 PDF Expert")
st.write("Upload PDFs, ask questions, and get expert answers powered by GPT.")

# Input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Model selection for OpenAI
model_options = ["gpt-4o-mini", "gpt-4"]
selected_model = st.selectbox("Select a model:", model_options)

# Initialize the vector store only if the API key is provided
if api_key:
    try:
        # Set OpenAI API key
        openai.api_key = api_key

        # Validate the API key by making a test request
        openai.Model.list()  # This will raise an exception if the key is invalid

        # Initialize the vector store
        vectorstore = initialize_vectorstore(api_key)

        # Process PDF upload
        uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            st.write("Processing documents...")

            try:
                documents = process_pdfs(uploaded_files)
                if documents:
                    # Add documents to the vector store
                    vectorstore.add_documents(documents)
                    vectorstore.persist()
                    st.write(f"Uploaded and processed {len(documents)} documents into the knowledge base.")
                else:
                    st.warning("No valid content found in the uploaded PDFs.")
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")

        # Chat interface
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Ask a question about the content in your PDFs:")
        if user_input:
            response_placeholder = st.empty()  # Placeholder for streaming response
            response_text = ""  # To accumulate streamed responses

            try:
                # Stream the response
                for chunk in get_chat_response(user_input, vectorstore, selected_model):
                    response_text += chunk  # Accumulate streamed text
                    response_placeholder.write(response_text)  # Update the UI with the new chunk

                st.session_state["chat_history"].append((user_input, response_text))
            except Exception as e:
                st.error(f"Error fetching response: {str(e)}")

        # Display chat history
        if st.session_state["chat_history"]:
            st.write("### Chat History")
            for i, (question, answer) in enumerate(st.session_state["chat_history"]):
                st.write(f"**Q{i+1}:** {question}")
                st.write(f"**A{i+1}:** {answer}")

    except openai.AuthenticationError:
        st.error("Invalid API key. Please check your API key and try again.")
    except openai.APIError as e:
        st.error(f"OpenAI API error: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

else:
    st.warning("Please enter your OpenAI API key to use the application.")
