from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader
import utils
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser

import requests
from collections import defaultdict
import re
import os
import json
from pymongo_get_database import get_database
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

def load_html(url_:list) -> list:
    loader = AsyncChromiumLoader([url_])
    html = loader.load()
    docs_transf = BeautifulSoupTransformer().transform_documents(html)
    return docs_transf[0].page_content

# 
def load_url_request(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    text = soup.get_text()
    return text


def vector_search_chroma(documents, query, k)->list:
    embedding_func = OpenAIEmbeddings()
    db = Chroma.from_documents(documents, embedding_func)
    docs = db.similarity_search(query,k=k)
    return docs#[doc.page_content for doc in docs]


def get_resume(file):
    pdf = PyMuPDFLoader(file).load()
    text = list()
    for page in range(len(pdf)):
        text.append(pdf[page].page_content)
    return '\n'.join(text)

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
def llm_query(context:str) -> str:    
    llm = Ollama(model="llama3")
    response = llm.invoke(context) 
    response_list = [r for r in response.split("\n") if len(r)>1]
    return ' '.join(response_list)


def load_txt(fname):
    f = open(fname, 'r')
    content = f.readlines()
    f.close()
    return content

def append_to_json(fname:str, data:dict):
    # Check if file exists
    if os.path.isfile(fname) is False:
        with open(fname, mode='w', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        with open(fname) as f:
            dictObj = json.load(f)
            dictObj.update(data)
        with open(fname, 'w') as json_file:
            json.dump(dictObj, json_file, 
                        indent=3,  
                        separators=(',',': '))
            

def update_data(save_dict, data):
    results = {
        "url": data[0],
        "question": data[1],
        "posting" : data[2],
        "response": data[3]
    }
    save_dict.append(results)
    return save_dict

def save_to_mongo(db_name:str, collection_name:str, data:dict):
    dbname = get_database(db_name)
    collection_name = dbname[collection_name]
    collection_name.insert_one(data)


def summarize_resume(docs, llm):
    
    # Define prompt
    prompt_template = PromptTemplate.from_template(
        ("Write a concise summary of the following:\\n\\n{context}")
    )
    # Invoke chain
    result = llm.invoke(prompt_template.format_prompt(context= docs))
    return result

def resume_to_posting_compare(llm, question, jobPost, resume):
    mistral = Ollama(model="mistral-nemo", temperature=0)
    # based on the next text: {} answer the question {}.

    job_template = PromptTemplate.from_template(
                    """You are an HR specialist. Answer the question {question} about the job posting {jobPost}
                    Respond concisely.
                    Return your answer as a bullet list.
                    """
                    )
    resume_sum = PromptTemplate.from_template(
                    """You are an HR specialist. Based on the resume {resume}, 
                    enumerate all skills of this candidate. 
                    Return your answer as a bullet list.
                    """
                    )
    response_template = PromptTemplate.from_template(
                    """
                    You are an HR specialist. Based on the candidate's skills {resume_sum}
                    and job requirements {requirements}, say if the candidate is suitable for the job.
                    Explain why for each bullet list. Respond concisely.
                    """
                    )
    job_chain = job_template | mistral | StrOutputParser()
    resume_chain = resume_sum | mistral | StrOutputParser()
    response_chain = response_template | mistral | StrOutputParser()
    # bool_chain = bool_resp | llm | StrOutputParser()
    chain = ({"requirements" : job_chain}  | {"resume_sum" : resume_chain} | response_chain )   
    # chain.max_tokens_limit = 100
    response = chain.invoke({"question":question,
                             "jobPost":jobPost,
                             "resume":resume
                             })
    return response