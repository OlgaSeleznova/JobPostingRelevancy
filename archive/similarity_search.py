from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import Chroma
import os
# import getpass
# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')



def get_relevant_chunk(documents:list, query:str)->list:
    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embedding_func)
    docs = db.similarity_search(query,k=3)
    return [doc.page_content for doc in docs]


def spacy_splitter(text:str) -> list:
    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def save_data(cont:str, outFile:str):
        f = open(outFile, "w")
        f.write(cont)
        f.close()


def main(file_name:str, outputFile:str):
    raw_documents = TextLoader(file_name).load()
    splitDocs = spacy_splitter(raw_documents[0].page_content)
    most_relevant = get_relevant_chunk(splitDocs, query="what are the job requirements?")
    save_data("\n\n".join(most_relevant), outputFile)
    

if __name__=="__main__":
    main(
        file_name = "posting/SeniorDataScientistPoeRemoteQuora.txt",
        outputFile = "chunks_extracted/SeniorDataScientistPoeRemoteQuora.txt"
    )