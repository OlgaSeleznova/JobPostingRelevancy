from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



def get_skills(docs, llm):
    # Define prompt
    resume_sum = PromptTemplate.from_template(
                """You are an HR specialist. Based on the resume {resume}, 
                extract only skills of the candidate. Return a list, each item in a newline. 
                Absolutely no introduction can be used. Use no digits or special characters.
                Answers should be concise. Do not include any explanations, introdictions or examples.  
                Never return introductions like "here is the list of skills"              
                For example:
                "Proficient in Python programming. \n
                Extensive experience with large sets of structured and unstructured data for machine learning, NLP, and speech recognition. \n
                Skilled in using ML frameworks such as PyTorch, HuggingFace, and Weights & Biases."
                """
                )
    # Invoke chain
    result = llm.invoke(resume_sum.format_prompt(resume=docs))
    return result


def summarize_cv(resume_docs, llm):
    # Define prompt
    resume_sum = PromptTemplate.from_template(
                """You are an HR specialist. Summarize the resume {resume}. 
                Absolutely no introduction can be used. Use no digits or special characters.
                Answers should be concise. Do not include any explanations, introdictions or examples.  
                Never return introductions like "here is the candidate's resume summary"              
                """
                )
    # Invoke chain
    result = llm.invoke(resume_sum.format_prompt(resume=resume_docs))
    return result


def get_requirements(docs, llm):
    # Define prompt
    resume_sum = PromptTemplate.from_template(
                """You are an HR specialist. Based on the job posting {jobPosting}, 
                extract only requirements for the position. Return a list, each item in a newline. 
                Absolutely no introduction can be used. Use no digits or special characters.
                Answers should be concise. Do not include any explanations, introdictions or examples.
                For example:
                "3+ years relevant work experience with building and automating data analytics and Machine Learning (ML) pipelines. \n
                Good understanding of ML and Al concepts. Hands-on experience in ML model development. \n
                Experience in operationalization of Machine Learning projects (MLOps) using at least one of the popular frameworks or platforms (e.g. Kubeflow, AWS Sagemaker, Google Al Platform, Azure Machine Learning, Databricks, DataRobot, MLFlow).
                """
                )
    # Invoke chain
    result = llm.invoke(resume_sum.format_prompt(jobPosting=docs))
    return result


def resume_to_posting_compare(llm, question, jobPost, resume):
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
                    You are an HR specialist. Based on the job requirements {requirements} 
                    and candidate's skills {resume_sum}. 
                    Enumerate requirements that match with the candidate's qualification.
                    Enumerate requirements that do not match.
                    Respond concisely and with bullet list.
                    """
                    )
    job_chain = job_template | llm | StrOutputParser()
    resume_chain = resume_sum | llm | StrOutputParser()
    response_chain = response_template | llm | StrOutputParser()
    chain = ({"requirements" : job_chain}  | {"resume_sum" : resume_chain} | response_chain )   
    response = chain.invoke({"question":question,
                             "jobPost":jobPost,
                             "resume":resume
                             })
    return response

def quesion_answering(require:str, skills:str, llm):
    questions_template = PromptTemplate.from_template("""You are an HR specialist. Based on the job posting requirement {require}, 
            generate a question for a candidate, that can be answered only YES or NO.
            Return only a question. No introductions or explanations can be used.
            For example,
            Reqirement: "Has expertise in NLP, safety, fairness, and Responsible AI."
            Questions: "Are you experienced in NLP, safety, fairness, and Responsible AI?"
            """)

    response_template = PromptTemplate.from_template("""
            You are a candidate, looking for a new job. Based on your skills {skills},
            answer question {questions}. You may not use any additional information except what is mentioned above.
            Answer only YES or NO. Be truthful.
            """)
    quest_chain = questions_template | llm | StrOutputParser()
    response_chain = response_template | llm | StrOutputParser()
    # response_template.format()
    # chain = ( {"questions":quest_chain} | response_chain)
    question = quest_chain.invoke({"require":require})
    response = response_chain.invoke({"skills":skills, "questions": question})
    return question, response


