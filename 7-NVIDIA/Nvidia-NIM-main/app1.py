import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# === Load environment variables ===
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# === App Title ===
st.title("🧠 Nvidia NIM Demo")

# === Initialize LLM ===
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# === Prompt Template ===
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# === Vector Embedding Function ===
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()

        st.write(f"📄 Loaded {len(st.session_state.docs)} PDF documents")

        if not st.session_state.docs:
            st.warning("⚠️ No documents found in ./us_census. Please check the folder and try again.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30]
        )

        st.write(f"🧩 Created {len(st.session_state.final_documents)} text chunks")

        if not st.session_state.final_documents:
            st.warning("⚠️ Document splitting failed. No chunks generated.")
            return

        # DEBUG: Show preview of chunks
        for i, doc in enumerate(st.session_state.final_documents[:2]):
            st.code(doc.page_content[:300])

        # Check if embedding works
        try:
            test_embedding = st.session_state.embeddings.embed_query("test")
            if not test_embedding:
                st.warning("❌ NVIDIA API key issue or embedding service failed.")
                return
        except Exception as e:
            st.error(f"Embedding test failed: {e}")
            return

        # Build FAISS vector DB
        try:
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            st.success("✅ FAISS vector store created!")
        except Exception as e:
            st.error(f"Vector store creation failed: {e}")
            return

# === UI: Embedding Trigger ===
if st.button("📥 Embed Documents"):
    vector_embedding()

# === UI: User Input ===
prompt1 = st.text_input("🔍 Enter Your Question from Documents")

# === Retrieval & Answering ===
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("❗ Please embed documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed = time.process_time() - start

        st.write("⏱️ Response Time:", round(elapsed, 2), "seconds")
        st.subheader("📌 Answer:")
        st.write(response.get('answer', "No answer found."))

        with st.expander("📄 Document Chunks Used"):
            for i, doc in enumerate(response.get("context", [])):
                st.markdown(doc.page_content)
                st.markdown("---")
