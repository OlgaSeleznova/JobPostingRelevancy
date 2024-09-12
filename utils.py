from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from bs4 import BeautifulSoup
from langchain_community.utils.math import cosine_similarity_top_k
import requests
import os
import json
from pymongo_get_database import get_database
from langchain_text_splitters import CharacterTextSplitter
from pandas import DataFrame


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
    return docs


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

def spacy_splitter(text:str, chunk:int, overlap:int) -> list:
    text_splitter = SpacyTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    chunks = text_splitter.create_documents([text])
    print(f"Number of chunks: {len(chunks)}")
    return chunks

# # filter and clean lines in a separate function
# def llm_query(context:str) -> str:    
#     llm = Ollama(model="llama3")
#     response = llm.invoke(context) 
#     response_list = [r for r in response.split("\n") if len(r)>1]
#     return ' '.join(response_list)


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


def conclude_responses(db_name, collection_name, llm):
    dbname = get_database(db_name)
    collection_name = dbname[collection_name]
    item_details = collection_name.find()
    responses = "\n\n".join([it["response"] for it in item_details])
    prompt = PromptTemplate.from_template(
                """
                You are an HR specialist. 
                Based on analysis of candidate's relevancy to job posting in responses {responses},  
                conclude if a candidate is suitable for the position. 
                Answer only YES or NO.
                """
                )
    result = llm.invoke(prompt.format_prompt(responses=responses))
    # TBD: save data
    return result

def collect_database_values(db_name, collection_name, unique_index):
    db = get_database(db_name)
    collection = db[collection_name]
    cursor = list(collection.find({"sessionID": unique_index}))
    df = DataFrame(cursor)

    return df


def character_split(text, chunk=100, overlap=0):
    text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=chunk,
                    chunk_overlap=overlap,
                    length_function=len,
                    is_separator_regex=False,
                )
    docs = text_splitter.create_documents([text])
    return docs

def find_most_similar(x,y):
    return cosine_similarity_top_k(x,y)

def format_docs(docs):
    return '\n'.join([doc.page_content for doc in docs])

def relevancy_metric(data):
    data = data.groupby("responses")[["requirement"]].apply(lambda x: x)#.reset_index().drop(columns="level_1")
    resp_match = data.loc["YES",:].values
    resp_miss = data.loc["NO",:].values
    return resp_match, resp_miss, int(resp_match.size/data.size*100)