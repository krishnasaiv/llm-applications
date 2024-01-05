import os
import json
from langchain.vectorstores import FAISS #Chroma
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
import pinecone, tiktoken
import streamlit as st
from streamlit_chat import message
from datetime import datetime

#vector stores
from langchain.vectorstores import Pinecone

#Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings

# Document Loaders
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader, WikipediaLoader

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLMs, Memory & Chains
from langchain.chat_models import ChatOpenAI
# from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

st.set_page_config(
    page_title='Your Custom Chatbot',
    # page_icon='🤖'
)
st.subheader('Welcome to your Custom Chatbot 🤖')
# st.session_state.vs = None
if 'history' not in st.session_state:
    st.session_state.history = []

# st.session_state.feedback = None



#################################
### Load Documents


def load_document(file_path):
    name, extension = os.path.splitext(file_path)
    
    if extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        loader = TextLoader(file_path)
    elif extension == '.md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        print(f"Unsupported File Format {file_path}. Supported formats: .pdf, .docx, .txt, .md")
        return []    
        
    print(f"Reading {file_path}")
    return loader.load()

### function that either takes a file path & reads the text from the file or a folder path & reads the text from each file 
def load_from(path, nested=False):
    ### If a file path is passed
    if os.path.isfile(path):
        return load_document(path)
    
    ### If a directory is passed
    elif os.path.isdir(path):
        print(f"Reading from folder {path}")
        item_paths = [os.path.join(path, f) for f in os.listdir(path)]
        
        loaded_docs = []
        for p in item_paths:
#             print(p)
            if os.path.isfile(p):
                loaded_docs += load_document(p)
            elif nested and os.path.isdir(p):
                loaded_docs += load_from(p)
        
        return loaded_docs

def load_from_wiki(query, load_max_docs=1, max_chars_per_doc=5000):
    loader = WikipediaLoader(query=query, load_max_docs=load_max_docs, doc_content_chars_max=max_chars_per_doc)
    data = loader.load()
    return data

#################################
### Chunk Data

def chunk_data(data, chunk_size=400, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap )
    chunks = text_splitter.split_documents(data)
    return chunks

#################################
### Create embeddings

def create_embeddings(chunks, openai_api_key):
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store=FAISS.from_documents(chunks, embeddings)
    return vector_store

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return f"Total Tokens: {total_tokens} |  Cost: {round(total_tokens / 1000 * 0.0004, 3)} $"


#################################
### Build LLM Chain

def get_llm_chain(vector_store, model_name, openai_api_key, num_chunks=5):
    # 1. Define LLM
    llm = ChatOpenAI(model=model_name, temperature=1, openai_api_key=openai_api_key)

    # 2. Vector Store retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': num_chunks})
    
    # 3. Define Chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    return chain
    

#################################
### Ask Questions
    
def ask(query, llm_chain, has_memory=False, chat_history = []):
    if not has_memory:
        answer = llm_chain.run(query)
        chat_history=[]
    else:
        chain_output = llm_chain({"question": query})
        answer, chat_history = chain_output['answer'], chain_output['chat_history']
    return answer, chat_history
        
#################################
### Build Functions to maintian / reset session
    
def reset_session():
    if 'history' in st.session_state:
        st.session_state['history'] = []

    # if 'vs' in st.session_state:
    #     del st.session_state['vs']



# def reset_vectorstore():
#     if 'vs' in st.session_state:
#         del st.session_state['vs']

#################################
### Main
   

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    with st.sidebar:
        api_key =  st.secrets["OPENAI_API_KEY"] #os.getenv('OPENAI_API_KEY') 

        
        uploaded_file = st.file_uploader("Upload a File: ", type=['pdf', 'docx', 'txt', 'md'], on_change=reset_session)
        model_name = "gpt-3.5-turbo"
        # model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4.0-turbo"], index=0, on_change=reset_session)
        chunk_size = st.number_input("Chunk Size: ", min_value=100, max_value=1024, value=512, on_change=reset_session)
        chunk_overlap = st.number_input("Chunk Overlap: ", min_value=20, max_value=256, value=80, on_change=reset_session)
        k = st.number_input("k: ", min_value=1, max_value=20, value=3, on_change=reset_session, )

        add_data = st.button("Load Data", on_click=reset_session)


        col1, col2 = st.columns([1,1])

        # with col1:
        #     if st.button('Reset Chat Session', on_click=reset_session):
        #         reset_session()
        # with col2:
        #     if st.download_button(label="Export Chat", data=json.dumps([str(message) for message in st.session_state.history], indent=4), file_name='chat_export.json', mime='application/json' ): 
        #         pass

        if uploaded_file and add_data:
            with st.spinner("Reading...", ):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                data = load_document(file_name)
                st.write("Reading... Done")

            with st.spinner("Chunking Data..."):
                chunks = chunk_data(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                st.write("Chunking Data... Done")
                st.write(f"Chunk size: {chunk_size}, Num Chunks: {len(chunks)}")

            with st.spinner("Create Embeddings..."):
                st.write(calculate_embedding_cost(chunks))
                vector_store = create_embeddings(chunks, openai_api_key=api_key) #insert_or_fetch_embeddings(index_name= index_name, chunks= chunks, embeddings_type=embeddings_type)
                
                st.write("Create/ Fetch Embeddings... Done")
            
            st.session_state.vs = vector_store
            st.success("file uploaded, chunked & embedded successfully")


    q = st.chat_input()
    if q:
        # if 'vs' in st.session_state:
        st.session_state.history.append(HumanMessage(content=q))

        vs = st.session_state.vs 

        llmchain = get_llm_chain(vector_store=vs , model_name=model_name, openai_api_key=api_key)
        with st.spinner('Working on your request ...'):
            answer, chat_history = ask(query=q, llm_chain=llmchain)
        st.session_state.history.append(AIMessage(content=answer))

                # st.text_area('LLM Answer: ', value=answer)
    with st.container():
        all_items = st.empty()  # This will be used to house our scrollable content

        # We'll use the column method to create a single column layout
        with all_items.container():
            for i, msg in enumerate(st.session_state.history):
                time_stamp = datetime.now().strftime("  %H:%M:%S")
                if i % 2 == 0:
                    message(f"{msg.content} {time_stamp}", is_user=True, key=f'{i} + 🤓') # user's question
                else:
                    message(msg.content, is_user=False, key=f'{i} +  🤖')



                # st.divider()

            # if 'chat_history' not in st.session_state:
            #     st.session_state.chat_history = ''

            # display_text = f'Question: {q} \nAns: {answer}'

            # st.session_state.chat_history = f'{display_text} \n {"-" * 100} \n {st.session_state.chat_history}'
            # hist = st.session_state.chat_history

            # st.text_area(label='Chat History', value=hist, key='chat_history', height=400)

            


        




