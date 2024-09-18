import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
import utils
import pandas as pd


def main():
    IFRAME = '<iframe src="https://github.com/OlgaSeleznova/Position_relevancy" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>'

    st.set_page_config(
        page_title="JobPostingRelevancy",
        page_icon=":clipboard:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    # st.markdown(
    #     f"""
    #     # Job Posting Relevance {IFRAME}
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.markdown(
        "# Welcome to Job Posting Relevance project! Check resume relevancy for the job posting."
    )

    uploaded_file = st.file_uploader("Load resume in PDF format", type="pdf")
    url = st.text_input("Paste URL link of the job posting", "")

    if uploaded_file is not None: 
        with open(uploaded_file.name, mode='wb') as w:
            w.write(uploaded_file.getvalue())
        if url is not None:
            jobPost, cv = utils.load_data(url_l=url, resume=uploaded_file.name)
            if cv and jobPost:
                
                llm = Ollama(model="llama3.1:70b", temperature=0)
                embedding_func = OpenAIEmbeddings()
                # extract job requirements from the posting
                requirement_docs = utils.extract_job_requirements(jobPost, embedding_func, llm)
                # generate cv vector store documents
                cv_vecs = utils.generate_cv_vecstore(cv, embed_func=embedding_func)
                #get position relevance and save data to mongo
                # indId = utils.save_relevance_mongo(requirement_docs, cv_vecs, llm, url, "job_postings_relevancy", 
                #                             "responses_collected")
                # all_responses = utils.collect_database_values(db_name="job_postings_relevancy", 
                #                         collection_name="final_relevancy",
                #                         unique_index=indId)

                #no mongo approach
                all_responses = utils.get_relevance_no_mongo(requirement_docs, cv_vecs,llm=llm)
                print(all_responses)
                resp_df = pd.DataFrame(list(zip(list(all_responses.values()), [req.page_content for req in requirement_docs])), 
                                       columns=["responses","requirement"])
                print(resp_df)
                if resp_df is not None:
                    st.markdown("## Results")
                    resp_match, resp_miss, result = utils.relevancy_metric(resp_df)
                    print(resp_match, resp_miss, result)
                    rescontainer = st.container(border=True)
                    rescontainer.write(f"Resume is {result} % relevant  for this job posting")

                    container1 = st.container(border=True)
                    container1.write("Matching requirements:")
                    container1.write(resp_match)

                    container2 = st.container(border=True)
                    container2.write("Missing requirements: ")
                    container2.write(resp_miss)


if __name__=="__main__":
    main()

#"https://lumenalta.com/jobs/ai-engineer"

    # main(
    #     url=url,
    #     resumeFname=uploaded_file,
    #     model_name="llama3.1:70b",
    #     response_collection_name="responses_collected",
    #     relevancy_collection_name="final_relevancy",
    #     db_name = "job_postings_relevancy"
    # )