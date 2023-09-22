from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
import json
import base64
from datetime import datetime
import streamlit as st
from streamlit_chat import message


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

st.set_page_config(
    page_title='You Custom Assistant',
    page_icon='🤖'
)
st.subheader('Welcome to your Custom ChatGPT 🤖')

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
st.session_state.vs = None

def reset_vectorstore():
    if 'history' in st.session_state:
        del st.session_state['history']

# creating the messages (chat history) in the Streamlit session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

st.session_state.feedback = None

# creating the sidebar
with st.sidebar:
    model_name = st.selectbox("Select Model", ["gpt-3.5-turbo", "gpt-4.0-turbo"])
    chat = ChatOpenAI(model_name=model_name, temperature=0.5)
    system_message = st.text_input(label='System role')

    # with st.expander("Upload Files"):
    #     uploaded_file = st.file_uploader("Upload a File: ", type=['pdf', 'docx', 'txt', 'md'])
    #     chunk_size = st.number_input("Chunk Size: ", min_value=100, max_value=1024, value=512)
    #     k = st.number_input("k: ", min_value=1, max_value=20, value=3)


    if system_message:
        if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
            st.session_state.messages.append(
                SystemMessage(content=system_message)
                )
            

            
    col1, col2 = st.columns([1,1])

    with col1:
        if st.button('Reset Chat'):
            st.session_state.messages = []
    with col2:
        if st.download_button(label="Export Chat", data=json.dumps([str(message) for message in st.session_state.messages], indent=4), file_name='chat_export.json', mime='application/json' ): 
            pass


if len(st.session_state.messages) >= 1:
    if not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))

print("--------------------------------\n")
print(st.session_state.messages)



user_prompt =  st.chat_input()


if user_prompt:
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    with st.spinner('Working on your request ...'):
        response = chat(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))



for i, msg in enumerate(st.session_state.messages[1:]):
    time_stamp = datetime.now().strftime("  %H:%M:%S")
    if i % 2 == 0:
        message(f"{msg.content} {time_stamp}", is_user=True, key=f'{i} + 🤓') # user's question
    else:
        message(msg.content, is_user=False, key=f'{i} +  🤖') # ChatGPT response



if st.session_state.messages and isinstance(st.session_state.messages[-1], AIMessage):  
    col3, col4, col5 = st.columns([1,1,1])

    with col3:
        st.text("Did you like the response?")
    with col4:
        if st.button('👍'):
            st.session_state.feedback = "Pos"
    with col5:
        if st.button('👎'):
            st.session_state.feedback = "Neg"
            st.session_state.messages.append(AIMessage(content="We are sorry to hear that. Could you please tell us why the response was not helpful?"))




if st.session_state.feedback:
    if st.session_state.feedback == 'Pos':
        message("Thank you for your positive feedback!", is_user=False, key=f'Positive +  🤖')
    else:
        message("We are sorry to hear that. Could you please tell us why the response was not helpful?", is_user=False, key=f'Negative +  🤖')
    st.session_state.feedback = None
