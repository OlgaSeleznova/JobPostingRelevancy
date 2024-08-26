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
from langchain_community.utils.math import cosine_similarity_top_k
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

import requests
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from collections import defaultdict
import re
import os
import json
import datetime
import prompts



# check length of the list before indexing!
def main(url, questionsFname, resumeFname, model_name, collect_name):
    llm = Ollama(model="llama3.1", temperature=0)
    # load data
    jobPost = utils.load_html(url)
    questions = TextLoader(questionsFname).load()[0].page_content.split("\n")
    pdf = PyMuPDFLoader(resumeFname).load()
    # get skills
    skills = prompts.get_skills(pdf,llm)
    skills_docs = utils.character_split(skills)
    # get job requirements
    requirements = prompts.get_requirements(jobPost, llm)
    requirement_docs = utils.character_split(requirements)
    # Get similarity
    # similar = utils.cosine_similarity(requirement_docs, skills_docs, top_k=3)
    # print(similar)

    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(skills_docs, embedding_func)
    for requirement in requirement_docs:
        docs = db.similarity_search(requirement.page_content, k=3)
        
        utils.save_to_mongo(db_name="similarities", 
                            collection_name=collect_name+datetime.datetime.now().strftime("%H:%M:%S_%d%m"),
                            data={
                                "url":url,
                                "requirement":requirement,
                                "mostSimilarDocs":utils.format_docs(docs)
                            })




    # resume_sum = utils.summarize_resume(docs=pdf, llm=llm)
    # get the most relevant parts of job posting, based in the resume summary
    # splitDocs = utils.spacy_splitter(text=text)
    # postingChunks = utils.vector_search_chroma(splitDocs, query=resume_sum, k=3)
    # mistral = Ollama(model=model_name, temperature=0)
    # collection_name=collect_name+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # get answers
    # for question in questions: 
        # run sequential chain per question
    #     answer = utils.resume_to_posting_compare(mistral, question, postingChunks[0].page_content, resume='\n'.join([r.page_content for r in pdf]))
    #     utils.save_to_mongo("job_posting_to_resume", collection_name, data={
    #                                                             "url": url,
    #                                                             "question": question,
    #                                                             "response": answer
    #                                                             })
    # result = utils.conclude_responses(db_name="job_posting_to_resume", collection_name=collection_name, llm=llm)
    # final_result = ""
    # if "yes" in result.lower():
    #     final_result == "Match"
    # if "no" in result.lower():
    #     final_result == "No match"
    # print(f"Final result is {result}")
    # utils.save_to_mongo("job_posting_to_resume", 
    #                     "verdicts", 
    #                     data={"url":url,
    #                         "verdict":final_result+"\n" +result,
    #                         "collection":collection_name,
    #                         "date":datetime.datetime.now().strftime("%H:%M:%S_%d%m")
    # })
 
if __name__=="__main__":
    main(
        url="https://lumenalta.com/jobs/ai-engineer",
        questionsFname="questions.txt", 
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        model_name="llama3.1",
        collect_name="lumenata_ai-eng"
    )
