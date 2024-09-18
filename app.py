import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
import utils
# from streamlit.web.cli import main
# st.write("Hello world")


# check length of the list before indexing!
def main(url, resumeFname, model_name, db_name, response_collection_name, relevancy_collection_name):
    llm = Ollama(model=model_name, temperature=0)
    embedding_func = OpenAIEmbeddings()
    # load data
    job_post = utils.load_html(url)
    resume = utils.format_docs(resumeFname)
    # extract job requirements from the posting
    requirement_docs = utils.extract_job_requirements(job_post, embedding_func, llm)
    # generate cv vector store documents
    cv_vecs = utils.generate_cv_vecstore(resume, embed_func=embedding_func)
    #get position relevance and save data to mongo
    indId = utils.save_relevance_mongo(requirement_docs, cv_vecs, llm, url, db_name, response_collection_name)
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
    # print(f"Resume is {result}% relevant for the position")


if __name__=="__main__":
    IFRAME = '<iframe src="https://github.com/OlgaSeleznova/Position_relevancy" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>'

    st.set_page_config(
        page_title="JobPostingRelevancy",
        page_icon=":clipboard:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(
        f"""
        # Job Posting Relevance {IFRAME}
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "Welcome to Job Posting Relevance project! Check resume relevancy for the job posting."
    )

    uploaded_file = st.file_uploader("Load resume in pdf format", type=["pdf"])
    url = st.text_input("Paste URL link of the job posting", "JOb posting url")

    main(
        url=url,
        resumeFname=uploaded_file,
        model_name="llama3.1:70b",
        response_collection_name="responses_collected",
        relevancy_collection_name="final_relevancy",
        db_name = "job_postings_relevancy"
    )