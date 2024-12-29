# Chatbot Project

This is a chatbot that works in a browser. It uses:
- **Streamlit** for the interface.
- **Google Generative AI (GenAI)** to generate responses.
- **ChromaDB** to store questions and answers.

## Installation

1. Clone this project:
   ```bash
   git clone https://github.com/231748/chatbot_project.git
   cd chatbot_project
   ```

2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the chatbot using Streamlit:
```bash
streamlit run app.py
```

Visit the link it shows (usually http://localhost:8501) in your browser to chat with the bot.

## Example Interaction

```
User: What is AI?
Bot: AI stands for Artificial Intelligence. It refers to systems designed to mimic human intelligence.
```

## Project Structure

```
chatbot_project/
├── README.md           # Project information
├── LICENSE             # License details
├── requirements.txt    # List of required libraries
├── src/                # Main chatbot program
├── test/               # Tests
└── chroma_db/          # Database folder
```

## Features

- **Chat Interface**: Easy-to-use chat feature built with Streamlit.
- **AI Responses**: Powered by Google GenAI for generating answers.
- **Query Storage**: Saves user questions and bot answers in ChromaDB.
