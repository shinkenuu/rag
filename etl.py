import logging
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.embeddings.openai import OpenAIEmbeddings


from constants import WEAVIATE_URL

_FILE_PATHS = "./textbooks/Data_Science_for_Business.pdf"

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 20
_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    "\t",
    "",
]
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=_SEPARATORS,
)


def extract(file_path: str):
    loader = PyPDFLoader(file_path)
    logging.info(f"Reading {file_path}")
    pages = loader.load()
    logging.info(f"Read {file_path}")

    for page in pages:
        logging.debug(page.page_content)
        if page.page_content:
            yield page


def transform(pages):
    clean_pages = []

    for page in pages:
        page.page_content = page.page_content.replace("\t", " ")
        clean_pages.append(page)

    documents = _SPLITTER.split_documents(clean_pages)
    return documents


def load(transformed_documents: list[str]):
    embeddings = OpenAIEmbeddings()
    db = Weaviate.from_documents(
        transformed_documents, embeddings, weaviate_url=WEAVIATE_URL, by_text=False
    )
    return db


def etl(file_path=None, transformer=transform):
    documents = extract(file_path)
    transformed_documents = transformer(documents)
    db = load(transformed_documents)
    return db


if __name__ == "__main__":
    file_path = _FILE_PATHS
    db = etl(file_path=file_path)

    query = "How can I cluster my customers?"
    docs = db.similarity_search(query)

    for doc in docs:
        print(doc.page_content)
