from llama_index.llms.ollama import Ollama
from llama_index.core import SummaryIndex
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import SimpleWebPageReader
import os
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
import re
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import SpacyTextSplitter

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



def main_llamaindex(url):
    llm = Ollama(model="llama3")
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])
    index = SummaryIndex.from_documents(documents)
    # set Logging to DEBUG for more detailed outputs
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("What are the job requirements?")
    print(response)


def load_html(url_):
    loader = AsyncChromiumLoader(url_)
    html = loader.load()
    headers = [get_header(h.page_content) for h in html]
    return html, headers


def html_transform(my_html, tags_to_extract):
    bs_transformer = BeautifulSoupTransformer()
    docs_transf = bs_transformer.transform_documents(my_html)
    return docs_transf


def get_header(htmlTxt):
    title = re.search(r'<title>(.*?)</title>', htmlTxt).group(1)
    return re.sub(r'\W+', '', title)

def get_digit(mystr):
    digits = list(filter(lambda x: x.isdigit(), list(mystr)))
    if len(digits)>0:
        return int(digits[0])
    else:
        return 0

def fileExists(fileName):
    if os.path.exists(fileName):
        dig = get_digit(fileName)
        fileName = fileName.split(".txt")[0].removesuffix(str(dig)) + str(dig+1) +".txt"
        if os.path.exists(fileName):
            return fileExists(fileName)
        else:
            return fileName
    else:
        return fileName

def save_data(cont, fname):
        outFile = fileExists(fname)
        f = open(outFile, "w")
        f.write(cont)
        f.close()

def main(url, outDir):
    html, headers = load_html(url)
    docs = html_transform(html, tags_to_extract=["span", ""])
    for ind in range(len(docs)):
        content = docs[ind].page_content
        print(f"Content length is {len(content)}")
        fileName = os.path.join(outDir,headers[ind]+".txt")
        save_data(content, fileName)



if __name__=="__main__":
    main(
        url=[
            # "https://www.accenture.com/ca-en/careers/jobdetails?id=R00218148_en&title=Responsible%20AI%20Engineer%20SpecialistD",
             "https://jobs.ashbyhq.com/quora/71e1b41f-ce2f-4144-a3b4-00c7d40f4e44?locationId=d03c8cce-ce51-4c89-9d06-ad8597402fc0"],
        outDir = "posting/"
    )