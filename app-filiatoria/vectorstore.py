from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

PDFS = [
    "data/Titulo_IV_Parentesco.pdf",
    "data/Titulo_V_Filiacion.pdf",
    "data/Titulo_VI_Adopcion.pdf",
    "data/Titulo_VII_Responsabilidad_Parental.pdf",
    "data/Titulo_VIII_Procesos_Familia.pdf",
]

def build_vectorstore():
    docs = []
    for pdf in PDFS:
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
