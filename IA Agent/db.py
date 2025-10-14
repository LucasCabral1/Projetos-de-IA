import os
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_BASE = os.path.join(BASE_DIR, "base")

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    documentos = []

    loaders = [
        PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf"),
        DirectoryLoader(PASTA_BASE, glob="*.txt", loader_cls=TextLoader),
        DirectoryLoader(PASTA_BASE, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader),
        DirectoryLoader(PASTA_BASE, glob="*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(PASTA_BASE, glob="*.csv", loader_cls=CSVLoader),
    ]

    for loader in loaders:
        try:
            documentos.extend(loader.load())
            print(f"Documentos carregados com sucesso de: {loader.__class__.__name__}")
        except Exception as e:
            print(f"Atenção: Não foi possível carregar documentos do tipo {loader.__class__.__name__}. Erro: {e}")
            continue

    print(f"Documentos carregados: {len(documentos)}")
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    print("Banco de Dados criado")

criar_db()