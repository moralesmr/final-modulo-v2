import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

#PDFs del Código
PDFS = [
    os.path.join(DATA_DIR, "Titulo_IV_Parentesco.pdf"),
    os.path.join(DATA_DIR, "Titulo_V_Filiacion.pdf"),
    os.path.join(DATA_DIR, "Titulo_VI_Adopcion.pdf"),
    os.path.join(DATA_DIR, "Titulo_VII_Responsabilidad_Parental.pdf"),
    os.path.join(DATA_DIR, "Titulo_VIII_Procesos_Familia.pdf"),
]

def build_vectorstore():
    docs = []

    # validacion
    for pdf in PDFS:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"No se encontró el PDF: {pdf}")

        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./ccyc_vectordb"
    )

    return vectorstore
