import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
import utils


def run_the_app(myResume, jobPosting, url):
    db_name="job_postings_relevancy"
    response_collection_name="responses_collected"
    llm = Ollama(model="llama3.1:70b", temperature=0)
    embedding_func = OpenAIEmbeddings()
    # extract job requirements from the posting
    requirement_docs = utils.extract_job_requirements(jobPosting, embedding_func, llm)
    # generate cv vector store documents
    cv_vecs = utils.generate_cv_vecstore(myResume, embed_func=embedding_func)
    #get position relevance and save data to mongo
    sessId = utils.save_relevance_mongo(requirement_docs, cv_vecs, llm, url, db_name, 
                                response_collection_name)
    return sessId


def main():
    st.set_page_config(
    page_title="JobPostingRelevancy",
    page_icon=":clipboard:",
    layout="wide",
    initial_sidebar_state="auto",
)
        
    st.title("🦜🔗 Job Posting Relevance ")
    st.markdown("Welcome to Job Posting Relevance project! Check resume relevancy for the job posting.")


    # SETTINGS
    with st.sidebar:
        st.markdown("#### Provide your resume in PDF format and a URL job posting below")
        uploaded_file = st.file_uploader("Load resume", type="pdf")
        if uploaded_file is not None: 
            with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())
        url_l = st.text_input("Paste URL link of the job posting", "")
    

    # RUN LLM
    if uploaded_file is not None and url_l is not None:
        jobPost, resume  = utils.load_data(url_l, uploaded_file.name)
        # resume=uploaded_file.name
        if resume and jobPost:
            db_name="job_postings_relevancy"
            response_collection_name="responses_collected"
            indId = run_the_app(resume, jobPost, url_l)
            all_responses = utils.collect_database_values(db_name=db_name, 
                            collection_name=response_collection_name,
                            unique_index=indId)
            # resp_df = pd.DataFrame(list(zip(list(all_responses.values()), [req.page_content for req in requirement_docs])), 
            #                         columns=["responses","requirement"])
            # if resp_df is not None:
            st.markdown("## Results")
            resp_match, resp_miss, result = utils.relevancy_metric(all_responses)
            st.markdown(f" #### Resume is {result} % relevant  for this job posting")

            container1 = st.container(border=True)
            container1.write("*Matching requirements:*")
            container1.write(resp_match)

            container2 = st.container(border=True)
            container2.write("*Missing requirements:*")
            container2.write(resp_miss)


if __name__=="__main__":
    main()
