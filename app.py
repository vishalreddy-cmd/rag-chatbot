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

st.set_page_config(page_title="Ask Vishal's AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .title-block {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .title-block h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .title-block p {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 0;
    }
    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.75rem 0;
    }
    div.stButton > button {
        width: 100%;
        background-color: #f0f4ff;
        color: #1a1a2e;
        border: 1.5px solid #d0d8f0;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 500;
        text-align: left;
        transition: all 0.2s ease;
        line-height: 1.4;
        height: auto;
        white-space: normal;
    }
    div.stButton > button:hover {
        background-color: #4f46e5;
        color: white;
        border-color: #4f46e5;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,70,229,0.25);
    }
    div.stButton > button:active {
        transform: translateY(0px);
    }
    .divider {
        border: none;
        border-top: 1px solid #eee;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-block">
    <h1>🤖 Ask Vishal's AI</h1>
    <p>Powered by LLaMA 3.1 · RAG · FAISS &nbsp;|&nbsp; Ask me anything about Vishal's experience, skills, and projects.</p>
</div>
""", unsafe_allow_html=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit secrets.")
    st.stop()

@st.cache_resource
def build_chain():
    loader = TextLoader("data/your_document.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant representing Vishal's professional profile.
    Answer the question based only on the context below.
    Be confident, concise, and professional.

    Context: {context}

    Question: {question}

    Answer:""")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = build_chain()

# Quick questions
st.markdown('<div class="section-label">✨ Quick Questions — Click to Ask</div>', unsafe_allow_html=True)

questions = [
    ("💼", "Where is Vishal currently working?"),
    ("🧠", "What is Vishal's most impressive project?"),
    ("🤖", "What AI and ML tools does Vishal know?"),
    ("✅", "Is Vishal a strong candidate for Data Science or AI roles?"),
    ("🚀", "Should I hire Vishal?"),
    ("🌍", "Is Vishal open to remote work?"),
]

selected_question = None
col1, col2 = st.columns(2)
for i, (icon, q) in enumerate(questions):
    col = col1 if i % 2 == 0 else col2
    if col.button(f"{icon}  {q}", key=f"q_{i}"):
        selected_question = q

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
query = st.chat_input("Or type your own question about Vishal...") or selected_question

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(query)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})