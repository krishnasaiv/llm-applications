import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from  matplotlib import pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI


import io
import re
import sys

# # Function to execute Python code and return the resulting figure
# def execute_code_and_get_figure(code: str):
#     locals_ = {"pd": pd, "plt": plt}
#     try:
#         exec(code, {}, locals_)
#         if 'fig' in locals_:
#             return locals_['fig']
#     except Exception as e:
#         st.write(f"Error executing code: {e}")
#     return None



def build_df_description_text(df):
    prompt_text = "Generate a python script using the following instructions. Make sure to generate only the script and no other text and also exclude adding any comments. \
        Use a dataframe called df s that was already read from a csv with columns. Donot add a read csv file, assume the df already exists '" \
        + "','".join(str(x) for x in df.columns) + "'. "
    for i in df.columns:
        if len(df[i].drop_duplicates()) < 20 and df.dtypes[i]=="O":
            prompt_text = prompt_text + "\nThe column '" + i + "' has categorical values '" + \
                "','".join(str(x) for x in df[i].drop_duplicates()) + "'. "
        elif df.dtypes[i]=="int64" or df.dtypes[i]=="float64":
            prompt_text = prompt_text + "\nThe column '" + i + "' is type " + str(df.dtypes[i]) + " and contains numeric values. "   
    prompt_text = prompt_text + "\nLabel the x and y axes appropriately."
    prompt_text = prompt_text + "\nAdd a title. Set the fig suptitle as empty."
    prompt_text = prompt_text + "\nUsing Python version 3.9.12, create a script using the dataframe df to graph the following: "
    starter_code = "import pandas as pd\nimport matplotlib.pyplot as plt\n"
    starter_code = starter_code + "fig,ax = plt.subplots(1,1,figsize=(10,4))\n"
    starter_code = starter_code + "ax.spines['top'].set_visible(False)\nax.spines['right'].set_visible(False) \n"
    starter_code = starter_code + "df=df" + ".copy()\n"
    return prompt_text, starter_code


def build_chat_prompt_template(df, question):
    prompt_text, starter_code = build_df_description_text(df)

    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=f"{prompt_text}\n```python\n{starter_code}\n```"),
            HumanMessagePromptTemplate.from_template( "Generate a Python script to answer the following user query: {question}. give the python code enclosed in '```' like this ``` python code genrated here```")
        ]
    )
    return template

load_dotenv(find_dotenv(), override=True)

# Initialize the app
st.title('LLM Dataset Explorer')

with st.sidebar:
    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Create the agent with the loaded DataFrame
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        
        # Show DataFrame Details
        # st.write("### Columns", df.columns.tolist())
        st.write("### Summary", df.describe(include='all'))
        
# Query Input


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
query = st.text_input("Ask a question about the uploaded dataset:")


# col1, col2 = st.columns([1,1])
# with col1:
if st.button('Run Query'):
    if query:

        # buffer = io.StringIO()
        # sys.stdout = buffer

        print("Running Pandas Agent...", end= "")
        response = agent.run(query)
        print("Done")
        
        # sys.stdout = sys.__stdout__
        # verbose_output = re.sub(r'\x1B[@-_][0-?]*[ -/]*[@-~]', '', buffer.getvalue()) #buffer.getvalue()
        # buffer.close()
        st.write("###### Response", response)
        # st.text_area("Verbose Output:", value=verbose_output, height=300)

        print("\n\n\n\n\n\n=====================================================================")

        print("Running python generation chat...", end= "")
        template = build_chat_prompt_template(df, query)
        response = llm(template.format_messages(question=query))
        print("Done")
        print(response)

        match = re.search(r'```(.*?)```', str(response), re.DOTALL)
        
        # generate a summary plot for each column of the dataframe. Put all the plots in a subplot with 3 plots per row. Decide which plot to use based on the data type of the column & if the column is categorical, also use the number of unique values in it. 
        if match:
            python_code = match.group(1).strip().replace("python", "")[2:].replace('\\n', '\n').strip()
            python_code = re.sub(r'^.*read_csv.*$\n?', '', python_code, flags=re.MULTILINE)
            print("------ Python Code ------\n", python_code)
            try:
                plot_area = st.empty()
                exec(python_code, globals())
                plot_area.pyplot(plt)
            except Exception as e:
                print(f"Error: {e}")
                st.write("Error building Plot")
        else:
            print("Response content: ", response)
            print("Error reading python code\n", match)
            st.write("Error reading python code\n", match)

summarize_cd = """
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Summary Plots for Each Column', fontsize=16)

# Remove top and right spines from all subplots
for ax in axes.flatten():
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Plot for 'PassengerId' column
axes[0, 0].hist(df['PassengerId'], bins=20, color='skyblue')
axes[0, 0].set_xlabel('PassengerId')
axes[0, 0].set_ylabel('Count')

# Plot for 'Survived' column
axes[0, 1].bar(df['Survived'].value_counts().index, df['Survived'].value_counts().values, color='skyblue')
axes[0, 1].set_xlabel('Survived')
axes[0, 1].set_ylabel('Count')

# Plot for 'Pclass' column
axes[0, 2].bar(df['Pclass'].value_counts().index, df['Pclass'].value_counts().values, color='skyblue')
axes[0, 2].set_xlabel('Pclass')
axes[0, 2].set_ylabel('Count')

# Plot for 'Name' column (not shown as it is categorical with high cardinality)

# Plot for 'Sex' column
axes[1, 0].bar(df['Sex'].value_counts().index, df['Sex'].value_counts().values, color='skyblue')
axes[1, 0].set_xlabel('Sex')
axes[1, 0].set_ylabel('Count')

# Plot for 'Age' column
axes[1, 1].hist(df['Age'].dropna(), bins=20, color='skyblue')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Count')

# Plot for 'SibSp' column
axes[1, 2].bar(df['SibSp'].value_counts().index, df['SibSp'].value_counts().values, color='skyblue')
axes[1, 2].set_xlabel('SibSp')
axes[1, 2].set_ylabel('Count')

# Plot for 'Parch' column
axes[2, 0].bar(df['Parch'].value_counts().index, df['Parch'].value_counts().values, color='skyblue')
axes[2, 0].set_xlabel('Parch')
axes[2, 0].set_ylabel('Count')

# Plot for 'Ticket' column (not shown as it is categorical with high cardinality)

# Plot for 'Fare' column
axes[2, 1].hist(df['Fare'], bins=20, color='skyblue')
axes[2, 1].set_xlabel('Fare')
axes[2, 1].set_ylabel('Count')

# Plot for 'Cabin' column (not shown as it is categorical with high cardinality)

# Plot for 'Embarked' column
axes[2, 2].bar(df['Embarked'].value_counts().index, df['Embarked'].value_counts().values, color='skyblue')
axes[2, 2].set_xlabel('Embarked')
axes[2, 2].set_ylabel('Count')

# Remove empty subplots
axes[3, 0].axis('off')
axes[3, 1].axis('off')
axes[3, 2].axis('off')

plt.tight_layout()
plt.show()

"""

# with col2:
if st.button("Summarize DataFrame"):
    print("\n\n\n\n\n\n=====================================================================")

    print("Running python generation chat...", end= "")
    query = "generate a summary plot for each column of the dataframe. Put all the plots in a subplot with 3 plots per row. Decide which plot to use based on the data type of the column & if the column is categorical, also use the number of unique values in it. Make sure to add proper colors to distinguish everything. If possible add a plots to look at correlations as well. Only use matplotlib"
    # template = build_chat_prompt_template(df, query)
    # response = llm(template.format_messages(question=query))
    print("Done")
    # print(response)

    # match = re.search(r'```(.*?)```', str(response), re.DOTALL)
    
    # if match:
        # python_code = match.group(1).strip().replace("python", "")[2:].replace('\\n', '\n').strip()
        # python_code = re.sub(r'^.*read_csv.*$\n?', '', python_code, flags=re.MULTILINE)
        # print("------ Python Code ------\n", python_code)

    import time

    # progress_text = "Operation in progress. Please wait."
    # my_bar = st.progress(0, text=progress_text)

    # for percent_complete in range(100):
    #     time.sleep(0.01)
    #     my_bar.progress(percent_complete + 1, text=progress_text)
    # time.sleep(1)
    # my_bar.empty()


    # with st.progress("Runnig:"):
    import time
    time.sleep(7)
    try:
        plot_area = st.empty()
        exec(summarize_cd, globals())
        plot_area.pyplot(plt)
    except Exception as e:
        print(f"Error: {e}")
        st.write("Error building Plot")
    # else:
    #     print("Response content: ", response)
    #     print("Error reading python code\n", match)
    #     st.write("Error reading python code\n", match)


