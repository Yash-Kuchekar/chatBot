import streamlit as st
from langchain_classic.chains.question_answering import load_qa_chain
#from streamlit import sidebar
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

OpenAI_API_KEY = "your-API-key"
st.header("My ChatBot")

with st.sidebar:
    st.title("My ChatBot")
    file = st.file_uploader("Upload PDF and start asking Questions",type="pdf")
#Extracting the Text from the File
if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text += page.extract_text()
        st.write(text)

    #Now Break it into the chinks
    splitter = RecursiveCharacterTextSplitter(separators=["\n"],chunk_size=200,chunk_overlap=50)
    chunks=splitter.split_text(text)
    st.write(chunks)
    #Creating object of OpenAIEmbedding class thst let us connect with open AI's Embeddings Model
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    #creating vector DB & storing embedding into it
    vector_store=FAISS.from_texts(chunks,embeddings)
    #FAISS is a DataBase

    #get user Query
    user_query = st.text_input("Type your Query Here")

    #Semantic Search from vector Store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        #Define LLm
        llm = ChatOpenAI(
            api_key = OpenAI_API_KEY,
            max_tokens = 300,
            temperature = 0,
            model = "gpt-3.5-turbo"
        )

        #Generating Response
        chain = load_qa_chain(llm,chain_type="stuff")
        output = chain.run(question=user_query,input_documents=matching_chunks)
        st.write(output)
