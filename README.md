# Chatbot Project

This project is a chatbot application that integrates Streamlit for the user interface, Google Generative AI (GenAI) for natural language processing, and ChromaDB for vector storage of queries and responses.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd chatbot_project
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the chatbot application using Streamlit:
```bash
streamlit run app.py
```

Open the link in your browser (usually http://localhost:8501) to interact with the chatbot.

## Examples

Example interaction:
```
User: What is AI?
Bot: AI stands for Artificial Intelligence. It refers to systems designed to mimic human intelligence.
```

## Project Structure

```
chatbot_project/
├── README.md           # Project documentation
├── LICENSE             # License file (optional)
├── requirements.txt    # Python dependencies
├── app.py              # Main application script
├── src/                # Source code files (if any additional are added)
├── test/               # Test scripts (future use)
├── chroma_db/          # Vector database folder
└── venv/               # Virtual environment folder (optional)
```

## Features

- **Chat Functionality**: Chat with the bot using a user-friendly interface powered by Streamlit.
- **LLM Integration**: Google Generative AI (Gemini) processes and generates responses.
- **Vector Storage**: Stores user queries and responses in ChromaDB.