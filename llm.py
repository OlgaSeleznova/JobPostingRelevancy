import os
from langchain_community.llms import Ollama

#function to load examples of phrases for few shot learning
def read_file(file):
    with open(file, 'r') as f:
        data = f.read().splitlines() 
    return data


#function to generate phrases for the user
def llm(questions_fname, resume_fname):
    llm = Ollama(model="llama3")
    questions = read_file(file=questions_fname)
    context = f"You are a helpful assistant. You answer the questions about an open job posting. You answer concisely and don't hallucinate."
    resume = read_file(file=resume_fname)
    
    for question in questions: 
        response = llm.invoke(context+question+"in the following job posting."+' '.join(resume)) 

    return response

llm(
    questions_fname="questions.txt", 
    resume_fname='chunks_extracted/SeniorDataScientistPoeRemoteQuora.txt'
)