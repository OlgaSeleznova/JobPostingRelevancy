from llama_index.llms.ollama import Ollama
from llama_index.core import SummaryIndex
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import SimpleWebPageReader
import os
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
import re
from langchain_community.document_transformers import BeautifulSoupTransformer


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
            "https://corp.flipp.com/careers/?gh_jid=5952833",
            "https://careers.spglobal.com/jobs/298823?lang=en-us",
            "https://www.grammarly.com/jobs/engineering/researcher-strategic-research?gh_jid=5829498",
            "https://www.grammarly.com/jobs/engineering/machine-learning-engineer-responsible-ai?gh_jid=5932771",
            "https://job-boards.greenhouse.io/gitlab/jobs/7522926002"
        ],
        outDir = "posting/"
    )