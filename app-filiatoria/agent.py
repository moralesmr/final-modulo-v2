from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

SYSTEM_PROMPT = """
Eres un agente jurídico especializado exclusivamente en Derecho de Familia argentino.

Reglas:
- Prioriza el Código Civil y Comercial.
- No inventes normas.
- Usa herramientas solo cuando corresponda.
"""

def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0)
    memory = MemorySaver()

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )
