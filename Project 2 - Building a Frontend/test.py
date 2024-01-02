import os
from langchain.vectorstores import Chroma
import pinecone, tiktoken
import streamlit as st

#vector stores
from langchain.vectorstores import Pinecone

#Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Document Loaders
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, WikipediaLoader

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLMs, Memory & Chains
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# 1. Define LLM
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

vector_store = None
# 2. Vector Store retriever
retriever = None if vector_store is None else vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# 3. Define Chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    x  = input("exter text")

    if x=='q':
        break
    
    
    if not vector_store:
        # If vector_store is present, use the chain
        response = llm(x)
        
    else:
        # If vector_store is not present, call the llm directly
        response = chain({"question": x})
        


