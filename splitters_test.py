
# SPLITTERS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import SpacyTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import en_core_web_sm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def recursive_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=0)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def semantic_split(text:str):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks
# nlp = en_core_web_sm.load()

def semantic_split(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def char_splitter(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False)
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks



def spacy_splitter(text):
    text_splitter = SpacyTextSplitter(chunk_size=1000)
    chunks = text_splitter.split_text(text)
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def main(file_name, outputFile):

    f = open(file_name, "r")
    content = f.read()
    # chunks = semantic_split(content)
    # chunks = char_splitter(content)
    chunks = recursive_splitter(content)
    # chunks = spacy_splitter(content)
    f = open(outputFile, "w")
    f.write("\n\n".join(chunks))
    f.close()


if __name__=="__main__":
    main(
        file_name = "posting/SeniorDataScientistPoeRemoteQuora.txt",
        outputFile = "chunks_extracted/SeniorDataScientistPoeRemoteQuora_recursive_splitter.txt"
    )