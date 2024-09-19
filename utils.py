from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import SpacyTextSplitter
from langchain_community.vectorstores import Chroma
from pymongo_get_database import get_database
from langchain_text_splitters import CharacterTextSplitter
from pandas import DataFrame
from langchain_community.vectorstores import Chroma
import datetime
import utils
import prompts
import uuid
import bson
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data(url_l, resume):
    job_post = utils.load_html(url_l)
    # pdf_ = PyMuPDFLoader(resume).load()
    # resume = utils.format_docs(pdf_)
    return job_post, resume


def extract_job_requirements(posting, embed_func, llm):
    # relevant chunk of data to improve performance because of issues
    jobPost_split = utils.spacy_splitter(posting, chunk=500, overlap=50)
    db_jp = Chroma.from_documents(jobPost_split, embed_func)
    jobPost_docs = db_jp.similarity_search("list of requirements", k=5)
    jp_requirements = utils.format_docs(jobPost_docs)
    requirements = prompts.get_requirements(jp_requirements, llm)
    docs = utils.character_split(requirements)
    return docs


def generate_cv_vecstore(resume, embed_func):
    resume_docs = utils.spacy_splitter(resume, chunk=200, overlap=5)
    # vectorize resume
    db_cv = Chroma.from_documents(resume_docs, embed_func)
    return db_cv


def extract_responses(requirement, resume, llm):
    docs = resume.similarity_search(requirement, k=4)
    mostSimSkills = utils.format_docs(docs)
    question, response = prompts.quesion_answering(requirement, mostSimSkills, llm=llm)
    return question, response


def parallel_process(requirements_cleaned, resume, llm, max_workers=4):
    results = []
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        future_to_req = {executor.submit(extract_responses, req, resume, llm): req for req in requirements_cleaned}
        # Gather the results as tasks are completed
        for future in as_completed(future_to_req):
            try:
                quest, resp = future.result()
                results.append((quest, resp))
            except Exception as exc:
                print(f'Error occurred: {exc}')
    return results


# def get_relevance_no_mongo(requirements, resume, llm) -> dict:
#     requirements_cleaned = [req.page_content for req in requirements]
#     results = parallel_process(requirements_cleaned, resume, llm, max_workers=4)
#     responce_collect = dict()
#     for i in range(len(results)):
#         question, response = results[i]
#         # generate dataframe with response and requirements
#         responce_collect[question] = response
#     return responce_collect


def save_relevance_mongo(requirements, resume, llm, url, db, collection):
    dbname = get_database(db)
    collection_name = dbname[collection]
    timestamp = datetime.datetime.now()
    sessionID = bson.Binary.from_uuid(uuid.uuid4())
    requirements_cleaned = [req.page_content for req in requirements]
    results = parallel_process(requirements_cleaned, resume, llm, max_workers=4)
    for i in range(len(results)):
        question, response = results[i]
        collection_name.insert_one({
                            "_id":bson.Binary.from_uuid(uuid.uuid4()),
                            "sessionID":sessionID,
                            "url":url,
                            "date":timestamp,
                            "requirement":requirements_cleaned[i],
                            "questions": question,
                            "responses": response
                        }) 
    return sessionID


def load_html(url:list) -> list:
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    docs_transf = BeautifulSoupTransformer().transform_documents(html)
    return docs_transf[0].page_content

# 
# def load_url_request(url):
#     page = requests.get(url)
#     soup = BeautifulSoup(page.content, "html.parser")
#     text = soup.get_text()
#     return text


# def vector_search_chroma(documents, query, k)->list:
#     embedding_func = OpenAIEmbeddings()
#     db = Chroma.from_documents(documents, embedding_func)
#     docs = db.similarity_search(query,k=k)
#     return docs


# def get_resume(file):
#     pdf = PyMuPDFLoader(file).load()
#     text = list()
#     for page in range(len(pdf)):
#         text.append(pdf[page].page_content)
#     return '\n'.join(text)

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

# def load_data(url_l, resume):
#     job_post = utils.load_html(url_l)
#     pdf_ = PyMuPDFLoader(resume).load()
#     resume = utils.format_docs(pdf_)
#     return job_post, resume

# def extract_job_requirements(posting, embed_func, llm):
#     # relevant chunk of data to improve performance because of issues
#     jobPost_split = utils.spacy_splitter(posting, chunk=500, overlap=50)
#     db_jp = Chroma.from_documents(jobPost_split, embed_func)
#     jobPost_docs = db_jp.similarity_search("list of requirements", k=5)
#     jp_requirements = utils.format_docs(jobPost_docs)
#     requirements = prompts.get_requirements(jp_requirements, llm)
#     docs = utils.character_split(requirements)
#     return docs


# def generate_cv_vecstore(resume, embed_func):
#     resume_docs = utils.spacy_splitter(resume, chunk=200, overlap=5)
#     # vectorize resume
#     db_cv = Chroma.from_documents(resume_docs, embed_func)
#     return db_cv


# def extract_responses(requirement, resume, llm):
#     docs = resume.similarity_search(requirement, k=4)
#     mostSimSkills = utils.format_docs(docs)
#     question, response = prompts.quesion_answering(requirement, mostSimSkills, llm=llm)
#     return question, response


# def parallel_process(requirements_cleaned, resume, llm, max_workers=4):
#     results = []
#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Submit tasks to the executor
#         future_to_req = {executor.submit(extract_responses, req, resume, llm): req for req in requirements_cleaned}
#         # Gather the results as tasks are completed
#         for future in as_completed(future_to_req):
#             try:
#                 quest, resp = future.result()
#                 results.append((quest, resp))
#             except Exception as exc:
#                 print(f'Error occurred: {exc}')
#     return results


def save_to_mongo(db_name:str, collection_name:str, data:dict):
    dbname = get_database(db_name)
    collection_name = dbname[collection_name]
    collection_name.insert_one(data)


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


def format_docs(docs):
    return '\n'.join([doc.page_content for doc in docs])


def relevancy_metric(data):
    uniqResp = data["responses"].unique()
    dataGR = data.groupby("responses")[["requirement"]].apply(lambda x: x)
    if "YES" and "NO" in uniqResp:
        resp_match = dataGR.loc["YES",:].values.flatten() 
        resp_miss = dataGR.loc["NO",:].values.flatten() 
    elif "YES" not in uniqResp:
        resp_match = "No matching values"
        resp_miss = dataGR.loc["NO",:].values.flatten() 
    elif "NO" not in uniqResp:
        resp_match = dataGR.loc["YES",:].values.flatten() 
        resp_miss = "No missing values"
    return resp_match, resp_miss, int(resp_match.size/dataGR.size*100)