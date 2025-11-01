import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key not found. Please set it in your environment or .env file.")
    st.stop()

st.header("📄 My ChatBot")

with st.sidebar:
    st.title("My ChatBot")
    file = st.file_uploader("Upload PDF and start asking questions", type="pdf")

# Extract text from the PDF
if file is not None:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""

    st.write("✅ PDF text extracted successfully!")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=200, chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    st.write(f"✅ Split into {len(chunks)} chunks")

    # Create embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User query input
    user_query = st.text_input("Type your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # Define the LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        # Run the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        output = chain.run(question=user_query, input_documents=matching_chunks)

        st.write("### 🤖 Answer:")
        st.write(output)
