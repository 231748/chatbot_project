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

# Load the Kazakhstan Constitution into memory
constitution_file = "kazakhstan_constitution.txt"
try:
    with open(constitution_file, "r", encoding="utf-8") as file:
        constitution_text = file.read()
except FileNotFoundError:
    raise Exception(f"Error: {constitution_file} not found. Ensure it is bundled with the program.")

# Add the constitution to the ChromaDB collection during initialization
collection.add(
    documents=[constitution_text],
    metadatas=[{"source": "Kazakhstan Constitution"}],
    ids=["constitution"]
)

# Streamlit app setup
st.title("Kazakhstan Constitution Chatbot")

# User input for chat
st.subheader("Ask Questions")
user_input = st.text_input("You:", "")

if user_input:
    try:
        # Query the collection for relevant context
        results = collection.query(query_texts=[user_input], n_results=3)

        # Flatten nested lists if results exist
        documents = [doc for sublist in results.get('documents', []) for doc in sublist]
        context = " ".join(documents) if documents else "No relevant context found."

        # Generate response using GenAI
        model_name = "gemini-1.5-pro"  # Specify the appropriate model name
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