import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from chromadb.errors import ChromaError

# Configure GenAI API
genai.configure(api_key="AIzaSyBidvoteLFoBFwhlcuHogb_sdqZVTu3scw")

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
st.title("Chatbot")

# User input
user_input = st.text_input("You:", "")

if user_input:
    try:
        # Query the collection for relevant context
        results = collection.query(query_texts=[user_input], n_results=1)

        # Flatten nested lists if results exist
        documents = [doc for sublist in results.get('documents', []) for doc in sublist]
        metadatas = [meta for sublist in results.get('metadatas', []) for meta in sublist]

        # Retrieve context if available
        context = " ".join(documents) if documents else ""

        # Generate response using GenAI
        model_name = "gemini-1.5-pro"
        model = genai.GenerativeModel(model_name)
        prompt = f"Context: {context}\nUser: {user_input}\nBot:"
        response = model.generate_content(prompt)
        bot_response = response.text if hasattr(response, 'text') else "I'm sorry, I couldn't generate a response."

        # Display the bot's response
        st.write(bot_response)

        # Save user input and bot response to the collection
        collection.add(
            documents=[user_input],
            metadatas=[{"response": bot_response}],
            ids=[str(hash(user_input))]
        )

    except ChromaError as ce:
        st.write(f"A ChromaDB error occurred: {ce}")
    except KeyError as ke:
        st.write(f"An error occurred while accessing data: {ke}")
    except ValueError as ve:
        st.write(f"An invalid value was encountered: {ve}")
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")