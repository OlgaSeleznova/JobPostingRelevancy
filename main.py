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
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
import utils
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
import re
import os
import json



# check length of the list before indexing!
def main(url, questionsFname, resumeFname, outputFname, postingParseContext, resumeParseContext):
    # load data
    text = utils.load_html(url)[0].page_content
    # text = utils.load_url_request(url=url)
    questions = TextLoader(questionsFname).load()[0].page_content.split("\n")
    resume = PyMuPDFLoader(resumeFname, headers=["Professional Summary", "Technical skills","Strengths"]).load()[0].page_content   
    
    splitDocs = utils.spacy_splitter(text=text)
    postingChunks = utils.vector_search_chroma(splitDocs, query=resume, k=3)
    responses = list()
    # get answers
    for question in questions: 
        postingResponse = utils.llm_query(postingParseContext + question+ ".".join(postingChunks))
        resumeAnswer = utils.llm_query(resumeParseContext + f"Statement: {postingResponse}. Resume: {resume}")
        utils.update_data(responses, (url, question, postingResponse, resumeAnswer))
    utils.append_to_json(fname=outputFname, data=responses)

if __name__=="__main__":
    main(
        url="https://lumenalta.com/jobs/ai-engineer",
        questionsFname="questions.txt", 
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        outputFname = "LLM-responses.json",
        postingParseContext = "You are a helpful assistant. You answer the questions about an open job posting. You answer concisely and do not hallucinate. If no information is mentioned response strictly 'not important'",
        resumeParseContext = "You are a helpful assistant. You check that the statement is correct for the resume. You answer strictly yes or no."
    )
