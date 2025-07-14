import os
import json
import boto3
import streamlit as st

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ——— AWS Bedrock setup ———
REGION = "us-east-1"
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

# Titan Embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

# ——— LLM constructors ———
def get_claude_llm():
    return ChatBedrock(
        model="us.anthropic.claude-opus-4-20250514-v1:0",
        client=bedrock_client,
        temperature=0.8,
        max_tokens=512,
    )

def get_llama2_llm():
    return ChatBedrock(
        model="us.meta.llama3-3-70b-instruct-v1:0",
        client=bedrock_client,
        temperature=0.8,
        max_tokens=512,
    )

# ——— Prompt Template ———
prompt_template = """
Human: Use the following pieces of context to provide a
concise answer to the question at the end. Summarize in at least
250 words with detailed explanations. If you don't know the answer,
just say so; don’t hallucinate.

<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ——— Build & Load FAISS index ———
def build_faiss_index():
    loader = PyPDFDirectoryLoader("data")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_documents(docs)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local("faiss_index")


def load_faiss_index():
    if not os.path.isdir("faiss_index"):
        raise FileNotFoundError(
            "Vector store not found—run 'Rebuild' in the sidebar first."
        )
    return FAISS.load_local(
        "faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

# ——— Retrieval QA ———
def answer_with(llm, index, query: str) -> str:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )
    return qa({"query": query})["result"]

# ——— Streamlit App ———
st.set_page_config(page_title="Chat with PDFs (Bedrock)")
st.title("📚 Chat with your PDFs via AWS Bedrock")

# Sidebar: rebuild vector store\
if st.sidebar.button("Rebuild Vector Store"):
    with st.spinner("Re‑building vector store…"):
        build_faiss_index()
        st.success("Vector store rebuilt!")

# Prompt for user question
if not os.path.exists("faiss_index"):
    st.info("No vector store found; click ‘Rebuild Vector Store’ in the sidebar.")

question = st.text_input("Ask a question about your PDFs:")

# Two model options
col1, col2 = st.columns(2)

with col1:
    if st.button("Claude Answer") and question:
        idx = load_faiss_index()
        llm = get_claude_llm()
        answer = answer_with(llm, idx, question)
        st.write(answer)

with col2:
    if st.button("LLaMA2 Answer") and question:
        idx = load_faiss_index()
        llm = get_llama2_llm()
        answer = answer_with(llm, idx, question)
        st.write(answer)

if __name__ == "__main__":
    # Streamlit runs main automatically
    pass
