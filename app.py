import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot — LLaMA 3.1 + FAISS")
st.markdown("Ask anything about the loaded document.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-actual-key-here")
if not GROQ_API_KEY or GROQ_API_KEY == "your-actual-key-here":
    raise RuntimeError("Missing GROQ_API_KEY. Set the GROQ_API_KEY environment variable before running the app.")

@st.cache_resource
def build_chain():
    # Load document
    loader = TextLoader("data/your_document.txt")
    docs = loader.load()

    # Chunk it
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # Embed and store in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLaMA 3.1 via Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the context below.
    
    Context: {context}
    
    Question: {question}
    
    Answer:""")

    # Modern RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about the document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(query)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})