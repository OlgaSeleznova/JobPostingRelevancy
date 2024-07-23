from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import SpacyTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import os
# import getpass
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')



def get_relevant_chunk(documents, query):
    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embedding_func)
    docs = db.similarity_search_with_score(query,k=3)
    return docs


def semantic_split(text):
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def spacy_splitter(text):
    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks

def recursive_splitter(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=0)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def main(file_name, outputFile):
    raw_documents = TextLoader(file_name).load()
    splitDocs = spacy_splitter(raw_documents[0].page_content)
    most_relevant = get_relevant_chunk(splitDocs, query="what are the job requirements?")
    # recursive_splitter(semantic_split)

if __name__=="__main__":

    main(
        file_name = "posting/SeniorDataScientistPoeRemoteQuora.txt",
        outputFile = "chunks_extracted/SeniorDataScientistPoeRemoteQuora_recursive_splitter.txt"
    )