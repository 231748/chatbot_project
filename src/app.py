import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st
import tiktoken

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader

OPENAI_API_KEY = "OPENAI_API_KEY"
openai.api_key = OPENAI_API_KEY


def fetch_constitution():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", class_="mt-5")
        if content_div:
            return "\n".join([p.text for p in content_div.find_all("p")])
        else:
            return "⚠️ Could not extract the Constitution text. The webpage structure may have changed."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Error fetching Constitution: {e}"


def find_relevant_section(user_query, document_text, max_tokens=4000):
    """Finds the most relevant section of the document to include in the context."""
    if not document_text:
        return "⚠️ No content available to search."

    sections = document_text.split("\n\n")  # Split by paragraphs
    best_match = max(sections, key=lambda sec: user_query.lower() in sec.lower(), default="")

    # Ensure best_match is not empty before tokenizing
    if not best_match.strip():
        return "⚠️ No relevant section found."

    # ✅ Check token length safely
    encoding = tiktoken.get_encoding("cl100k_base")
    while len(encoding.encode(best_match)) > max_tokens and len(best_match) > 50:
        best_match = best_match[:len(best_match) // 2]  # Reduce content

    return best_match


def ask_chatbot(user_query, document_content):
    """Generates a response using OpenAI GPT with retrieved context."""
    try:
        relevant_section = find_relevant_section(user_query, document_content)

        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in Kazakhstan Constitution."},
                {"role": "system", "content": f"Relevant section: {relevant_section}"},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI API Error: {e}"


st.title("Kazakhstan Constitution Chatbot")
st.markdown("*Fetching the latest version of the Constitution...*")

constitution_text = fetch_constitution()

if "⚠️" not in constitution_text:
    st.success("Constitution loaded successfully!")
else:
    st.error(constitution_text)


st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

docs = []
if uploaded_files:
    for file in uploaded_files:
        loader = PyPDFLoader(file)
        docs.extend(loader.load())

vector_store = None
retriever = None
multi_query_retriever = None

if docs:
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=OpenAI(model_name="gpt-4"))


def rag_response(user_query):
    """
    Generates multiple queries, retrieves relevant documents, and fuses them into an LLM answer.
    """
    if not vector_store:
        return "⚠️ No uploaded documents. Please upload PDFs to enable document-based queries."

    queries = multi_query_retriever.get_relevant_documents(user_query)
    docs_content = [doc.page_content for doc in queries]

    prompt_template = PromptTemplate.from_template(
        "Use the following documents to answer the question:\n{documents}\n\nQuestion: {query}\nAnswer:"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
    )

    response = qa_chain.run({"documents": docs_content, "query": user_query})
    return response


user_question = st.text_input("Ask a question about the Constitution or uploaded documents:")

if st.button("Submit"):
    if not user_question.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        if docs:
            answer = rag_response(user_question)
        else:
            answer = ask_chatbot(user_question, constitution_text)

        st.markdown(f"**Chatbot:** {answer}")