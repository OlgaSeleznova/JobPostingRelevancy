from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from collections import defaultdict
import re
import os
import json


def load_html(url_:list) -> list:
    loader = AsyncChromiumLoader([url_])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transf = bs_transformer.transform_documents(html)
    return docs_transf


def vector_search_chroma(documents, query, k)->list:
    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embedding_func)
    docs = db.similarity_search(query[0],k=k)
    if len(docs)>1:
        return [doc.page_content for doc in docs]
    else:
        return docs[0].page_content
    # return vectordb.as_retriever()

def vector_search_faiss(documents, query, k)->list:
    embedding_func = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embedding_func)
    docs = db.similarity_search(query[0],k=k)
    if len(docs)>1:
        return [doc.page_content for doc in docs]
    else:
        return docs[0].page_content
    

def html_transform(my_html:list, tags_to_extract):
    bs_transformer = BeautifulSoupTransformer()
    docs_transf = bs_transformer.transform_documents(my_html, tags_to_extract=["span", ""] )
    return docs_transf[0].page_content.split("\n")

def spacy_splitter(text:str) -> list:
    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks


def llm_query(context, posting, llm):    
    response = llm.invoke(context + '. '.join(posting)) 
    response_list = [r for r in response.split("\n") if len(r)>1]
    return response_list

# vector_search_chroma doesn't work!!!
def main(urlsFname,ind, questionsFname, resumeFname, outputFname, postingParseContext, resumeParseContext):
    # load data
    urls = TextLoader(urlsFname).load()
    currUrl = urls[0].page_content.split("\n")[ind]
    question_docs = TextLoader(questionsFname).load()
    resume = PyMuPDFLoader(resumeFname, headers=["Professional Summary", "Technical skills","Strengths"]).load()
    questions  = question_docs[0].page_content.split("\n")
    docs = load_html(currUrl)
    llm = Ollama(model="llama3")
    metadata = dict()
    # for doc in raw_docs:
        # for quest in questions:
    splitDocs = spacy_splitter(docs[0].page_content)
    postingChunks = vector_search_chroma(splitDocs, query="what are the job requirements?", k=3)
    responses = dict()
    # get answers
    for question in questions: 
        postingAnswer = llm_query(postingParseContext + question+"in the following job posting.", postingChunks, llm)
        # responses[question] = postingAnswer
        # ask about resume
        if "NaN" not in postingAnswer:
            resumeChunk = vector_search_faiss(resume, query=postingAnswer, k=1)
            resumeAnswer = [llm_query(resumeParseContext + f"Is {postingAnswer} mentioned in the following resume", resumeChunk, llm)]
            responses[question] = dict(zip(resumeAnswer,postingAnswer))
            metadata[docs.metadata["source"]] = responses
    json_data = json.dumps(metadata, indent=2)

    with open(outputFname, "a") as outfile:
        outfile.write(json_data)

if __name__=="__main__":
    main(
        urlsFname="position_urls.txt",
        ind = 0,
        questionsFname="questions.txt", 
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE.pdf",
        outputFname = "LLM-responses.json",
        postingParseContext = "You are a helpful assistant. You answer the questions about an open job posting. You answer concisely and do not hallucinate. If nothing is mentioned return strictly NaN.",
        resumeParseContext = "You are a helpful assistant. You compare You answer strictly yes or no. Do not hallucinate."
    )