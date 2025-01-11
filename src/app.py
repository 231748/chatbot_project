import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import ChromaError

# Configure GenAI API
genai.configure(api_key="AIzaSyBidvoteLFoBFwhlcuHogb_sdqZVTu3scw")  # Replace with your actual API key

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize embedding function
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Create or retrieve the collection
collection_name = "chatbot_memory"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

# Streamlit app setup
st.title("Chatbot with File and Memory")

# File upload section
st.subheader("Upload Files")
uploaded_files = st.file_uploader("Upload your files for analysis (supports .txt)", accept_multiple_files=True, type=["txt"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("utf-8")
        collection.add(
            documents=[file_content],
            metadatas=[{"filename": uploaded_file.name}],
            ids=[str(hash(uploaded_file.name))]
        )
    st.success(f"{len(uploaded_files)} file(s) uploaded and stored in memory.")

# User input for chat
st.subheader("Ask Questions or Chat")
user_input = st.text_input("You:", "")

if user_input:
    try:
        # Query the collection for relevant context
        results = collection.query(query_texts=[user_input], n_results=3)

        # Flatten nested lists if results exist
        documents = [doc for sublist in results.get('documents', []) for doc in sublist]
        context = " ".join(documents) if documents else "No relevant context found."

        # Generate response using GenAI
        model_name = "gemini-1.5-pro"
        model = genai.GenerativeModel(model_name)
        prompt = f"Context: {context}\nUser: {user_input}\nBot:"
        response = model.generate_content(prompt)
        bot_response = response.text if hasattr(response, 'text') else "I'm sorry, I couldn't generate a response."

        # Display the bot's response
        st.write(f"Bot: {bot_response}")

        # Save user input and bot response to the collection
        collection.add(
            documents=[user_input],
            metadatas=[{"response": bot_response}],
            ids=[str(hash(user_input))]
        )

    except ChromaError as ce:
        st.error(f"A ChromaDB error occurred: {ce}")
    except KeyError as ke:
        st.error(f"An error occurred while accessing data: {ke}")
    except ValueError as ve:
        st.error(f"An invalid value was encountered: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")