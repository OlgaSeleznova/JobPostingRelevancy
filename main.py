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
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from collections import defaultdict
import re
import os
import json



# check length of the list before indexing!
def main(url, questionsFname, resumeFname, outputFname, postingParseContext, resumeParseContext):
    llm = Ollama(model="llama3", temperature=0)
    # load data
    text = utils.load_html(url)
    questions = TextLoader(questionsFname).load()[0].page_content.split("\n")
    pdf = PyMuPDFLoader(resumeFname).load()
    # resume = utils.get_resume(resumeFname)
    resume_sum = utils.summarize_resume(docs=pdf, llm=llm)
    # get the most relevant parts of job posting, based in the resume summary
    splitDocs = utils.spacy_splitter(text=text)
    postingChunks = utils.vector_search_chroma(splitDocs, query=resume_sum, k=3)
    
    # get answers
    for question in questions: 
        # run sequential chain per question
        answer = utils.resume_to_posting_compare(llm, question, postingChunks[0].page_content, resume='\n'.join([r.page_content for r in pdf]))
        # postingResponse = utils.llm_query(postingParseContext + question+ ".".join(postingChunks[0].page_content))
        # resumeAnswer = utils.llm_query(resumeParseContext + f"Statement: {postingResponse}. Resume: {resume_sum}")
        utils.save_to_mongo("job_posting_to_resume", "mistral-nemo", data={
                                                                "url": url,
                                                                "question": question,
                                                                # "posting" : postingResponse,
                                                                "response": answer
                                                                })

 
if __name__=="__main__":
    main(
        url="https://lumenalta.com/jobs/ai-engineer",
        questionsFname="questions.txt", 
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        outputFname = "LLM-responses.json",
        postingParseContext = "You are a helpful assistant. You answer the questions about an open job posting. You answer concisely. You return answers only as a list, split with a newline.",
        resumeParseContext = "You are a helpful assistant. You check that the statement is correct for the resume. You answer strictly yes or no."
    )
