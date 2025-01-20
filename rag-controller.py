#1. Ingest Data from URL
#2. Split into chunks
#3. Send chunks to embeddings model
#4. Save embeddings to vector DB
#5. Perform similarity search
#6. Retrieve and display based on user query

import os
import ollama
#import streamlit as st
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

URLS = [
    'https://meghanthetravelingteacher.com/arizona-road-trip-itinerary/',
    'https://www.sprinkleofthis.com/amazing-arizona-road-trip/',
    'https://www.earthtrekkers.com/arizona-road-trip-itinerary/'    
]
model = 'llama3.2'

loader = WebBaseLoader(URLS)
data = loader.load()

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
urls_data = text_splitter.split_documents(data)

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=urls_data,
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

#Set up model
llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three
    different versions of the given user question to retrieve relevant documents from
    a vector database. Your goal is to help the user overcome some of the limitations 
    of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

rag = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

#prompt template
template = """Answer the question based on following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": rag, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


#Prompt
res = chain.invoke(input=("Plan a seven day road trip to Arizona",))

print(res)