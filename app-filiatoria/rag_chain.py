from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_classic.schema.output_parser import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""
Eres un asistente jurídico especializado en Derecho de Familia argentino.

CONTEXTO:
{context}

PREGUNTA:
{question}

INSTRUCCIONES OBLIGATORIAS:
- Respondé únicamente con información que surja del CONTEXTO.
- Indicá claramente si el tema está o no regulado por el Código Civil y Comercial.
- No inventes normas.
""")

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
