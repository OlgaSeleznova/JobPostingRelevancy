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
    docs = db.similarity_search(''.join(query),k=k)
    return [doc.page_content for doc in docs]


# def html_transform(my_html:list) -> list:
#     bs_transformer = BeautifulSoupTransformer()
#     docs_transf = bs_transformer.transform_documents(my_html, tags_to_extract=["span", ""] )
#     return docs_transf[0].page_content.split("\n")

def spacy_splitter(text:str) -> list:
    text_splitter = SpacyTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks

# filter and clean lines in a separate function
def llm_query(context, llm) -> str:    
    response = llm.invoke(context) 
    response_list = [r for r in response.split("\n") if len(r)>1]
    return ' '.join(response_list)


# check length of the list before indexing!
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
    splitDocs = spacy_splitter(docs[0].page_content)
    postingChunks = vector_search_chroma(splitDocs, query=questions, k=3)
    responses = dict()
    # get answers
    for question in questions: 
        postingAnswer = llm_query(postingParseContext + question+"in the following job posting."+ " ".join(postingChunks), llm)
        # responses[question] = postingAnswer
        # ask about resume
        # if "NaN" not in postingAnswer:
        resumeChunk = resume[0].page_content
        resumeAnswer = llm_query(resumeParseContext + f" Here is the relevant information: {postingAnswer}. Here is the resume: {resumeChunk}", llm)
        # responses[question] = dict(zip(resumeAnswer,postingAnswer))
        responses[question] = {resumeAnswer: postingAnswer}
    metadata[docs[0].metadata["source"]] = responses
    json_data = json.dumps(metadata, indent=3)

    with open(outputFname, "a") as outfile:
        outfile.write(json_data)

if __name__=="__main__":
    main(
        urlsFname="position_urls.txt",
        ind = 0,
        questionsFname="questions.txt", 
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        outputFname = "LLM-responses.json",
        postingParseContext = "You are a helpful assistant. You answer the questions about an open job posting. You answer concisely and do not hallucinate. If nothing is mentioned return strictly NaN.",
        resumeParseContext = "You are a helpful assistant. You answer strictly yes or no. Do not hallucinate. Is the following information mentioned in the resume? "
    )

    # https://careers.tiktok.com/position/7381563894341126409/detail DOESN'T WORK
