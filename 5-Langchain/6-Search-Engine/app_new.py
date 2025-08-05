import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
# You don't need dotenv if you are using st.secrets
# from dotenv import load_dotenv

# It's better to initialize wrappers and tools once
@st.cache_resource
def get_tools():
    ## Arxiv and wikipedia Tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    # This name is important for the agent to identify the tool
    search = DuckDuckGoSearchRun(name="duckduckgo_search")
    
    return [search, arxiv, wiki]

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

# Resolve literal “master” alias
if api_key.strip().lower() == "master":
    # api_key = "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"
    # api_key=st.secrets["GROQ_API_KEY"]
    api_key= secrets.GROQ_API_KEY

else:
    api_key = api_key.strip()

if not api_key:
    st.error("Please enter your Groq API key")
    st.stop()



# Initialize tools
tools = get_tools()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Accept user input
if prompt := st.chat_input(placeholder="What is the main cause of the French Revolution?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # The agent expects a string input, not the whole message history.
    # We will pass the latest user prompt to the agent.
    input_query = prompt

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    
    # The `initialize_agent` is deprecated. It's better to use newer agent creation methods,
    # but for this to work, we stick to the original code's logic.
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True # Set to True to see agent's thoughts in the terminal
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # Pass the string prompt to the agent, not the entire chat history
        response = search_agent.run(input_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)