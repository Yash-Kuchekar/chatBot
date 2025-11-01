import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os


os.environ["GROQ_API_KEY"] = "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"


# 🔹 Groq API key (you can get it free from https://console.groq.com)
GROQ_API_KEY = "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"

st.header("My ChatBot")

with st.sidebar:
    st.title("My ChatBot")
    file = st.file_uploader("Upload PDF and start asking Questions", type="pdf")

# Extract text from PDF
if file is not None:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Create embeddings (no API required)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user query
    user_query = st.text_input("Type your Query Here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # Use free Groq model instead of OpenAI
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="mixtral-8x7b-32768",  # Fast, free LLM
            temperature=0
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        output = chain.run(question=user_query, input_documents=matching_chunks)
        st.write(output)
