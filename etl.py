import logging
import torch
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

os.environ("OPENAI_SECRET_KEY", "")

_FILE_PATHS = (
    "media/Data_Science_for_Business.pdf"
)

### FOLLOW THIS
# https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476

#HUGGINGFACE
# _MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(
#     model_name=_MODEL_NAME, 
#     model_kwargs={
#        "device": "cuda:0" if torch.cuda.is_available() else "cpu"
#     }
# )


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

loader = TextLoader("path/to/file.pdf")
documents = loader.load()
docs = _SPLITTER.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

query = "Can you explain me how to use standard deviation?"
docs = db.similarity_search(query)

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


def parse_o_reilly_table_of_contents_line(line: str):
    if not line:
        return

    chapter = sub_chapter = page_number = None

    line = line.strip()

    if ". . " in line:
        chapter, *_, page_number = line.split(" .")
    elif "   " in line:
        sub_chapter, *_, page_number = line.split("  ")

    return chapter, sub_chapter, page_number


def transform_o_reilly(pages):
    clean_pages = []

    table_of_contents = []
    chapter = sub_chapter = page_number = None

    for page in pages:

        if "table of contents" in page.page_content.lower():
            for line in page.page_content.split("\n"):
                chapter_, sub_chapter_, page_number_ = parse_o_reilly_table_of_contents_line(line)

                chapter = chapter_ or chapter
                sub_chapter = sub_chapter_ or sub_chapter
                page_number = page_number_ or page_number

                if page_number:
                    table_of_contents.append((chapter, sub_chapter, page_number))
        
        latest_page_content = page.page_content.strip().split()[-1]
        if latest_page_content.isdigit():
            clean_pages.append(page)

    documents = _SPLITTER.split_documents(clean_pages)
    return documents


def load(transformed_documents: list[str]):
    db = FAISS.from_documents(documents=transformed_documents, embedding=embeddings)

    return db


def etl(file_path=None, transformer=transform):
    documents = extract(file_path)
    transformed_documents = transformer(documents)
    db = load(transformed_documents)
    return db


if __name__ == "__main__":
    file_path = _FILE_PATHS
    db = etl(file_path=file_path, transformer=transform_o_reilly)

    # file_path = _FILE_PATHS[-1]
    # db = etl(file_path=file_path)

    query = "the best way to lie with data"
    docs = db.similarity_search(query)

    for doc in docs:
        print(doc.page_content)
