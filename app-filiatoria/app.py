import streamlit as st
from langchain_core.messages import HumanMessage

from config import load_env
from vectorstore import build_vectorstore
from tools import build_tools
from agent import build_agent

st.set_page_config(page_title="Asistente Derecho de Familia 🇦🇷")

load_env()

@st.cache_resource
def init_agent():
    vectorstore = build_vectorstore()
    tools = build_tools(vectorstore)
    return build_agent(tools)

agent = init_agent()

st.title("⚖️ Asistente Jurídico – Derecho de Familia Argentino")

question = st.text_input("Ingresá tu consulta jurídica:")

if question:
    config = {"configurable": {"thread_id": "streamlit"}}

    with st.spinner("Analizando..."):
        for step in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config,
            stream_mode="values",
        ):
            respuesta = step["messages"][-1].content

    st.markdown(respuesta)
