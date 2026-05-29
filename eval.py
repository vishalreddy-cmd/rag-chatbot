import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your-actual-key-here")
if not GROQ_API_KEY or GROQ_API_KEY == "your-actual-key-here":
    raise RuntimeError("Missing GROQ_API_KEY. Set the GROQ_API_KEY environment variable before running eval.py.")

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
    You are a helpful assistant. Answer the question based only on the context below.
    Reply with the shortest possible answer — just the key fact, no full sentences.
    
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

def exact_match(predicted, expected):
    return int(predicted.strip().lower() == expected.strip().lower())

def f1_score(predicted, expected):
    pred_tokens = predicted.strip().lower().split()
    exp_tokens = expected.strip().lower().split()
    common = set(pred_tokens) & set(exp_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate():
    chain = build_chain()
    with open("data/eval_questions.json") as f:
        qa_pairs = json.load(f)

    total_em = 0
    total_f1 = 0

    print(f"\n{'Question':<50} {'Expected':<25} {'Got':<40} {'EM':<5} {'F1':<5}")
    print("-" * 130)

    for item in qa_pairs:
        question = item["question"]
        expected = item["expected"]
        predicted = chain.invoke(question).strip().rstrip(".")

        em = exact_match(predicted, expected)
        f1 = f1_score(predicted, expected)

        total_em += em
        total_f1 += f1

        print(f"{question:<50} {expected:<25} {predicted:<40} {em:<5} {round(f1,2):<5}")

    n = len(qa_pairs)
    print(f"\n✅ Exact Match: {round(total_em/n, 2)}")
    print(f"✅ F1 Score:    {round(total_f1/n, 2)}")

if __name__ == "__main__":
    evaluate()