# Kazakhstan Constitution Chatbot

This chatbot answers questions about the Constitution of Kazakhstan. It uses:
- **Streamlit**: To show the chatbot in a web browser.
- **Google Generative AI (GenAI)**: To create responses.
- **ChromaDB**: To save questions and answers.

## How to Install

1. Download the project:
   ```bash
   git clone https://github.com/231748/chatbot_project.git
   cd chatbot_project
   ```

2. Set up a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

Run the chatbot with this command:
```bash
streamlit run src/app.py
```

Open the link shown in your browser (usually http://localhost:8501) to start using the chatbot.

## Example Questions

### Example 1
**You**: What rights do citizens have?  
**Bot**: Citizens have many rights, such as:
- **Right to Life (Article 15)**: The death penalty is not allowed.
- **Freedom of Speech (Article 20)**: Censorship is not allowed.
- **Right to Privacy (Article 18)**: Your personal communications are protected.

### Example 2
**You**: What does Article 20 say?  
**Bot**: Article 20 protects freedom of speech and creativity. It says censorship is not allowed.

## Project Layout

```
chatbot_project/
├── README.md           # This file
├── LICENSE             # License information
├── requirements.txt    # List of libraries the project needs
├── src/                # The main chatbot code
├── test/               # Code for testing the project
└── chroma_db/          # Database files
```

## Features

- Answers questions based on Kazakhstan's Constitution.
- You can chat with it in your browser.
- Uses AI to provide clear and accurate answers.
- Keeps track of questions and answers using ChromaDB.