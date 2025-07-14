# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
# from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
# from langchain.agents import initialize_agent,AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# import os
# from dotenv import load_dotenv
# from langchain.tools import Tool
# from langchain_community.utilities import SerpAPIWrapper
# from duckduckgo_search import DDGS
# from duckduckgo_search import DDGS
# import random
# import time


# def duckduckgo_search_with_backoff(query: str, max_retries: int = 3) -> str:
#     """
#     Perform a DuckDuckGo search with retries and backoff on rate limits.
#     Returns the first result body or a friendly error message.
#     """
#     user_agents = [
#         # A small pool of common desktop UAs
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
#         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
#         "Mozilla/5.0 (X11; Linux x86_64)",
#     ]
#     headers = {"User-Agent": random.choice(user_agents)}
#     backoff = 1.0

#     for attempt in range(1, max_retries + 1):
#         try:
#             # throttle between attempts
#             if attempt > 1:
#                 time.sleep(backoff)
#                 backoff *= 2  # exponential backoff

#             with DDGS(headers=headers) as ddgs:
#                 results = ddgs.text(query)
#                 if results:
#                     # return the first snippet (body) we get
#                     return results[0].get("body", "No snippet available.")
#                 return "No results found."

#         except Exception as e:
#             err = str(e).lower()
#             # If it’s a rate-limit or HTTP error, retry; else break
#             if "ratelimit" in err or "429" in err or "rate limit" in err:
#                 continue
#             else:
#                 # non-rate-limit error: give up
#                 return f"⚠️ Search failed: {e}"

#     return "⚠️ DuckDuckGo rate‑limit exceeded, please try again later."

# # Wrap into a LangChain Tool
# search = Tool(
#     name="Search",
#     func=duckduckgo_search_with_backoff,
#     description="Useful for answering general questions via DuckDuckGo."
# )

# # use the below when all normal
# # search=DuckDuckGoSearchRun(name="Search")


# # search = Tool(
# #     name="Search",
# #     func=custom_ddg_search,
# #     description="Searches the web using DuckDuckGo"
# # )

# # search = Tool(
# #     name="Search",
# #     func=SerpAPIWrapper().run,
# #     description="Useful for answering questions about current events"
# # )

# ## Arxiv and wikipedia Tools
# arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

# api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
# wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# # search=DuckDuckGoSearchRun(name="Search")


# st.title("🔎 LangChain - Chat with search")
# """
# In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
# """

# ## Sidebar for settings
# st.sidebar.title("Settings")
# api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

# if "messages" not in st.session_state:
#     st.session_state["messages"]=[
#         {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
#     ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg['content'])

# if prompt:=st.chat_input(placeholder="What is machine learning?"):
#     st.session_state.messages.append({"role":"user","content":prompt})
#     st.chat_message("user").write(prompt)

#     llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
#     tools=[search,arxiv,wiki]

#     search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
# # AgentType.ZERO_SHOT_REACT_DESCRIPTION agent dont ely on chat history where as CHAT.ZERO_SHOT_REACT_DESCRIPTION does
#     with st.chat_message("assistant"):
#         st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#         # response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
#         try:
#             response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
#         except Exception as e:
#             response = f"⚠️ An error occurred: {str(e)}"

#         st.session_state.messages.append({'role':'assistant',"content":response})
#         st.write(response)






import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name="search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API Key:",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)
# # AgentType.ZERO_SHOT_REACT_DESCRIPTION agent dont ely on chat history where as CHAT.ZERO_SHOT_REACT_DESCRIPTION does

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)

