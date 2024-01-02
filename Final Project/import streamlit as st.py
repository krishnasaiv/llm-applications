import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
import sys
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from dotenv import load_dotenv, find_dotenv

# Function to execute Python code and return the resulting figure
def execute_code_and_get_figure(code: str):
    locals_ = {"pd": pd, "plt": plt}
    try:
        exec(code, {}, locals_)
        if 'fig' in locals_:
            return locals_['fig']
    except Exception as e:
        st.write(f"Error executing code: {e}")
    return None

# Modified Streamlit App
load_dotenv(find_dotenv(), override=True)
st.title('LLM Dataset Explorer')

# Sidebar for uploading CSV file
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        st.write("### Columns", df.columns.tolist())
        st.write("### Summary", df.describe())

# Main area for user input and displaying response
query = st.text_input("Ask a question about the uploaded dataset:")
if st.button('Run Query') and query:
    buffer = io.StringIO()
    sys.stdout = buffer
    response = agent.run(query)
    sys.stdout = sys.__stdout__
    verbose_output = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', buffer.getvalue())
    buffer.close()

    # Extract Python code from the response and execute it
    # Assume the Python code is in triple backticks, e.g., ```python code```
    match = re.search(r'```(.*?)```', response, re.DOTALL)
    if match:
        python_code = match.group(1).strip()
        fig = execute_code_and_get_figure(python_code)
        if fig:
            st.pyplot(fig)
        else:
            st.write("### Response", response)
            st.text_area("Verbose Output:", value=verbose_output, height=300)
    else:
        st.write("### Response", response)
        st.text_area("Verbose Output:", value=verbose_output, height=300)

