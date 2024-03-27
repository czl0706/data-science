# %%
import pandas as pd
df = pd.read_csv("submit.csv")

# %%
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-0125-preview")

# %%
print(f'Number of questions: {len(df)}')
print(f'Number of unique tasks: {len(df["task"].unique())}')

# %%
from langchain.prompts.prompt import PromptTemplate

template = '''
You are a high school student you are well prepared for the collge exam on {task}.

The following is a question that fits your subject:

{input}

A: {A}

B: {B}

C: {C}

D: {D}

In order to get the best score on the collge exam, you must choose the best option that makes your score higher.
It's very important to your future, your family will be proud of you if you get a good score.

Choose the best option and just answer with only one letter.
'''

prompt = PromptTemplate(input_variables=["input", "task", "A", "B", "C", "D", "example"], template=template)

chain = prompt | llm

# %%
from time import sleep

result = {'ID': [], 'target': []}

for index, row in df.iterrows():
    d = row.to_dict()
    
    d['task'] = d['task'].replace('_', ' ')    
    
    answer = chain.invoke(row.to_dict()).content
    
    final_answer = answer[0].upper()
    
    print(row.iloc[0], answer)
    
    result['ID'].append(row.iloc[0])
    result['target'].append(final_answer)
    sleep(1)

# %%
result_df = pd.DataFrame(result, columns=['ID', 'target'])

result_df.to_csv("answer.csv", index=False)

print("Dumped to answer.csv")


