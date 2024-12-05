'''
How this will work:
* Reading in some docs
* Oolama encode those docs
* Use in Memory index for RAG

TODO: write your own query ... chapter 8 tutorial 1
'''

import sys
import os
import pypdf 
import ollama

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


def pdf_pages(folder):
    pages = []
    for f in os.listdir(folder):
        if f.endswith('.pdf'):
            path = f"{folder}/{f}"
            reader = pypdf.PdfReader(path)
            for page in reader.pages:                
                pages.append(page.extract_text())
    return pages


def split(docs):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents(docs)
    return texts


def db(text):
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large:latest",
    )
    db = InMemoryVectorStore(embeddings)
    db.add_documents(text)
    return db, embeddings


def promt_template():
    return PromtTemplate(
        input_variables=['chunks_formatted', 'query'],
        template= """
        You are an exceptional customer support chatbot answering questions.
        
        You know the following information.

        {chunks_fromatted}

        Answer the following question from a customer. Use only information from the previous
        context information. Do not invent stuff.

        Question: {query}
        """
    )
    

def generate_answers(query, db, promt_template):
    llm = OllamaLLM(model='llama3.2:latest')    
    docs = db.similarity_search(query)
    retrieved_chunks = [doc.page_content for doc in docs]
    chunks_formatted = "\n\n".join(retrieved_docs)
    promt_formatted = promt_template.format(chunks_formatted=chunks_formatted, query=query)
    answer = llm(promt_formatted)
    return answer


if __name__ == '__main__':
    folder = sys.argv[1]
    query = sys.argv[2]
    pages = pdf_pages(folder)
    text = split(pages)
    retriever_db, embeddings = db(text)
    template = promt_template()
    answer = generate_answer(query, retriever_db, template)
    print(answer)
