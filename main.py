from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
import datetime
import utils
import prompts
import uuid
import bson
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor, as_completed



def load_data(url_l, resume):
    job_post = utils.load_html(url_l)
    pdf_ = PyMuPDFLoader(resume).load()
    resume = utils.format_docs(pdf_)
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


def save_relevance_mongo(requirements, resume, llm, url, db, collection):
    dbname = utils.get_database(db)
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


# check length of the list before indexing!
def main(url, resumeFname, model_name, db_name, response_collection_name, relevancy_collection_name):
    llm = Ollama(model=model_name, temperature=0)
    embedding_func = OpenAIEmbeddings()
    # load data
    jobPost, cv = load_data(url, resumeFname)
    # extract job requirements from the posting
    requirement_docs = extract_job_requirements(jobPost, embedding_func, llm)
    # generate cv vector store documents
    cv_vecs = generate_cv_vecstore(cv, embed_func=embedding_func)
    #get position relevance and save data to mongo
    indId = save_relevance_mongo(requirement_docs, cv_vecs, llm, url, db_name, response_collection_name)
    
    all_responses = utils.collect_database_values(db_name=db_name, 
                            collection_name=response_collection_name,
                            unique_index=indId)
    resp_match, resp_miss, result = utils.relevancy_metric(all_responses)
    utils.save_to_mongo(db_name=db_name, 
                            collection_name=relevancy_collection_name,
                            data={
                                "url":url,
                                "result":result,
                                "matching":'.'.join(resp_match.flatten().tolist()),
                                "missing":'.'.join(resp_miss.flatten().tolist())
                            })
    print(f"Resume is {result}% relevant for the position")
 

if __name__=="__main__":
    main(
        url="https://www.morganmckinley.com/jobs/ontario/machine-learning-engineer/1085188?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic",
        resumeFname="data/OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        model_name="llama3.1:70b",
        response_collection_name="responses_collected",
        relevancy_collection_name="final_relevancy",
        db_name = "job_postings_relevancy",
    )
