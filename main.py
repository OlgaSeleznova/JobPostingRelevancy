from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
import datetime
import utils
import prompts



# check length of the list before indexing!
def main(url, resumeFname, model_name, collection_name, db_name, final_db_name, jp_max_chunk=5000):
    llm = Ollama(model=model_name, temperature=0)
    embeddings = OpenAIEmbeddings()
    now = datetime.datetime.now().strftime("%H:%M:%S_%d%m")
    # load data
    jobPost = utils.load_html(url)
    if len(jobPost) > jp_max_chunk:
    # get job requirements
        # relevant chunk of data to improve performance because of issues
        jobPost_split = utils.spacy_splitter(jobPost, chunk=500, overlap=50)
        db_jp = Chroma.from_documents(jobPost_split, embeddings)
        jobPost_docs = db_jp.similarity_search("list of requirements", k=5)
        jp_requirements = utils.format_docs(jobPost_docs)
    else:
        jp_requirements = jobPost
    requirements = prompts.get_requirements(jp_requirements, llm)
    requirement_docs = utils.character_split(requirements)
    # get resume
    pdf = PyMuPDFLoader(resumeFname).load()
    resume = utils.format_docs(pdf)
    resume_docs = utils.spacy_splitter(resume, chunk=200, overlap=5)
    # vectorize resume
    db_cv = Chroma.from_documents(resume_docs, embeddings)
    collec_name_datetimed = collection_name+now
    for requireDocs in requirement_docs:
        requirement = requireDocs.page_content
        docs = db_cv.similarity_search(requirement, k=4)
        mostSimSkills = utils.format_docs(docs)
        question, response = prompts.quesion_answering(requirement, mostSimSkills, llm=llm)
        print(f"Saving data for requirement *{requirement}*")
        utils.save_to_mongo(db_name=db_name, 
                            collection_name=collec_name_datetimed,
                            data={
                                "url":url,
                                "requirement":requirement,
                                "mostSimilarDocs":mostSimSkills,
                                "questions":question,
                                "responses":response
                            })
    all_responses = utils.collect_database_values(db_name=db_name, 
                            collection_name=collec_name_datetimed,
                            value_to_extract="responses")
    result = utils.relevancy_metric(all_responses)
    utils.save_to_mongo(db_name=final_db_name, 
                            collection_name="collection",
                            data={
                                "url":url,
                                "result":result,
                                "collection": collection_name,
                                "model_name":model_name,
                                "datetime":now
                            })
    print(f"Resume is {result}% relevant for the position")
 

if __name__=="__main__":
    main(
        url="https://www.morganmckinley.com/jobs/ontario/machine-learning-engineer/1085188?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic",
        resumeFname="data/OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        model_name="llama3.1:70b",
        collection_name="morganmckinley-ml_eng",
        db_name = "cv_jp_collection",
        final_db_name = "cv_final_relevancy"
    )
