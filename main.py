from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import LLMChain
import datetime
import utils
import prompts



# check length of the list before indexing!
def main(url, resumeFname, model_name, collect_name):
    llm = Ollama(model=model_name, temperature=0)
    # load data
    jobPost = utils.load_html(url)
    pdf = PyMuPDFLoader(resumeFname).load()
    # get resume
    resume = utils.format_docs(pdf)
    resume_docs = utils.spacy_splitter(resume, chunk=200, overlap=5)
    # get job requirements
    requirements = prompts.get_requirements(jobPost, llm)
    requirement_docs = utils.character_split(requirements)
    # vectorize resume
    db = Chroma.from_documents(resume_docs, OpenAIEmbeddings())
    collection_name = collect_name+datetime.datetime.now().strftime("%H:%M:%S_%d%m")
    for requireDocs in requirement_docs:
        requirement = requireDocs.page_content
        docs = db.similarity_search(requirement, k=4)
        mostSimSkills = utils.format_docs(docs)
        question, response = prompts.quesion_answering(requirement, mostSimSkills, llm=llm)
        print(f"Saving data for requirement *{requirement}*")
        utils.save_to_mongo(db_name="similarities", 
                            collection_name=collection_name,
                            data={
                                "url":url,
                                "requirement":requirement,
                                "mostSimilarDocs":mostSimSkills,
                                "questions":question,
                                "responses":response
                            })
    

 
if __name__=="__main__":
    main(
        url="https://lumenalta.com/jobs/ai-engineer",
        resumeFname="OlgaSeleznova_MLEngineer_TEMPLATE (1).pdf",
        model_name="llama3.1:70b",
        collect_name="lumenata_ai-eng"
    )
