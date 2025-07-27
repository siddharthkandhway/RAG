import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio
import fitz  # PyMuPDF for PDF reading
import tempfile

# -------------------------------
# 🔑 LOAD API KEY FROM ENV
# -------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your system or .env file.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=api_key)

# -------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Chatbot with Gemini", layout="wide")
st.title("📚 Advanced RAG Chatbot using Gemini Pro")

# -------------------------------
# SESSION STATE (Chat History)
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# -------------------------------
# UI CONTROLS
# -------------------------------
st.sidebar.header("⚙ Settings")
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, step=50)
retrieval_k = st.sidebar.slider("Number of Chunks to Retrieve", 1, 10, 3, step=1)

model_choice = st.sidebar.selectbox("Choose Model", [
    "models/gemini-2.5-pro",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash"
])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

# Persistent FAISS folder
FAISS_PATH = "faiss_index"

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader("Upload documents (TXT or PDF)", type=['txt', 'pdf'], accept_multiple_files=True)

# -------------------------------
# PROCESS DOCUMENTS
# -------------------------------
if uploaded_files and st.button("Process Documents"):
    st.write("Processing documents...")
    documents = []

    for file in uploaded_files:
        if file.type == "text/plain":  # TXT file
            text = file.read().decode('utf-8', errors='ignore')
            documents.append(Document(page_content=text, metadata={"source": file.name}))
        elif file.type == "application/pdf":  # PDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(file.read())
                temp_pdf_path = temp_pdf.name
            with fitz.open(temp_pdf_path) as pdf:
                pdf_text = ""
                for page in pdf:
                    pdf_text += page.get_text()
            documents.append(Document(page_content=pdf_text, metadata={"source": file.name}))

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for doc in documents:
        for chunk in text_splitter.split_text(doc.page_content):
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))

    # Ensure event loop exists for gRPC
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Create or update FAISS vector store
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(all_chunks)
    else:
        vectorstore = FAISS.from_documents(all_chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_PATH)

    st.session_state["vectorstore"] = vectorstore
    st.success("✅ Documents processed and stored successfully!")

# -------------------------------
# LLM & RETRIEVER
# -------------------------------
if st.session_state["vectorstore"]:
    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": retrieval_k})
    llm = ChatGoogleGenerativeAI(model=model_choice, temperature=temperature)

    # -------------------------------
    # CHAT INTERFACE
    # -------------------------------
    query = st.text_input("Ask a question about the documents:")

    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            result = qa_chain.run(query)

            # Save to history
            st.session_state["history"].append({"question": query, "answer": result})

    # Display chat history
    st.write("### Chat History")
    for i, chat in enumerate(st.session_state["history"]):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
