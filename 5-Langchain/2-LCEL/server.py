# server.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes  # now imports the correct package

# 1. Load your env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY", "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")

# 2. Build your chain
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=api_key)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}:"),
    ("user", "{text}")
])
parser = StrOutputParser()
chain = prompt | model | parser

# 3. Create FastAPI & mount the chain
app = FastAPI(
    title="LangServe + Groq Demo",
    version="1.0",
    description="A simple translation API on /chain"
)
add_routes(app, chain, path="/chain")

# 4. Sanity‐check endpoint
@app.get("/ping")
def ping():
    return {"pong": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
