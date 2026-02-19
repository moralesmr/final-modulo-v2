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

#HISTORIAL
if "chat" not in st.session_state:
    st.session_state.chat = []

# Mostrar historial
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#INPUT
question = st.chat_input("Escribí tu consulta jurídica...")

if question:
    # 👤 Usuario
    st.session_state.chat.append({
        "role": "user",
        "content": f"👤 {question}"
    })

    with st.chat_message("user"):
        st.markdown(f"👤 {question}")

    config = {"configurable": {"thread_id": "streamlit"}}

    #Asistente
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            for step in agent.stream(
                {"messages": [HumanMessage(content=question)]},
                config,
                stream_mode="values",
            ):
                respuesta = step["messages"][-1].content

            st.markdown(f"🤖 {respuesta}")

    st.session_state.chat.append({
        "role": "assistant",
        "content": f"🤖 {respuesta}"
    })
