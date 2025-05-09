import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "API_KEY"  # Replace with your key

# --- Extract text from PDF ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join([page.extract_text() or "" for page in reader.pages])

# --- Setup Page ---
st.set_page_config(page_title="Chat with PDF", layout="centered")
st.title("💬 Chat with ChaturAI ")

# --- Session State Init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- Upload PDF ---
uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

# --- Process PDF if new upload ---
if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
    st.session_state.pdf_name = uploaded_file.name
    with st.spinner("🔍 Processing PDF & generating embeddings..."):
        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        # Split into chunks
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
        # LLM + QA Chain
        llm = ChatGroq(model_name="llama3-8b-8192")
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=False
        )
    st.success("✅ PDF processed! Ask your questions below.")

# --- Chat Input ---
if st.session_state.qa_chain:
    user_query = st.chat_input("💬 Ask something about the PDF...")
    if user_query:
        with st.spinner("🤖 Thinking..."):
            response = st.session_state.qa_chain.run(user_query)
        # Store chat
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("ai", response))

# --- Show Chat History ---
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)

# --- Show reminder if PDF not uploaded ---
if not st.session_state.qa_chain:
    st.info("📁 Please upload a PDF to begin.")
