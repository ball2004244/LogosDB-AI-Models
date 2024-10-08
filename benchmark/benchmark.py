'''
This file is for benchmarking the LogosDB as RAG for LLama 3.1 8B on MMLU dataset.
'''

import sys
import os
import time
# Add the absolute path to slm_tune to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from slm_tune.call import raw_call
from call_rag import call_rag
from datasets import load_dataset
import pandas as pd

subject = 'business_ethics'
ds = load_dataset("cais/mmlu", subject)

print('Keys:', ds.keys())


# convert the dataset to a pandas dataframe
df = pd.DataFrame(ds['test'])
print(f'Length of test set: {len(df)}')
print(f'Top 5 rows:\n{df.head()}')

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
PROMPT = '''
You are an expert in %s.
You are given a question and a list of choices. Choose the best answer from the choices given.
The initial choices come as a list of string, but the desired answer must be only "A", "B", "C", "D", or "E". 
No yapping. No explain. No nothing. Just the letter.

Here is your question:
%s

Here are the choices:
%s

Example answer:
A

Your answer:

'''

# now start benchmark on single SLM
# store all result in a file in benchmark/results
def benchmark_raw(df: pd.DataFrame) -> None:
    print('Starting benchmarking on raw SLM...')
    start = time.perf_counter()
    res_dir = 'benchmark/results'
    res_file = 'llama_raw.txt'
    res_path = os.path.join(res_dir, res_file)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    topic = df['subject'][0]
    with open(res_path, 'w') as f:
        f.write(f'Topic: {topic}\n')

    for i, row in df.iterrows():
        print(f'Processing row {i}/{len(df)}...')
        question = row['question']
        choices = row['choices']

        prompt = PROMPT % (subject, question, choices)

        raw_res = raw_call(prompt, model=OLLAMA_MODEL)
        
        # extract the final answer from the raw response
        res = raw_res.split('Final answer: ')[-1].strip()
        with open(res_path, 'a') as f:
            f.write(f'{res}\n')
            # f.write(f'Question: {question}\n')
            # f.write(f'Choices: {choices}\n')
            # f.write(f'Answer: {res}\n')
            # f.write('--------------------------\n')

    print(f'Benchmarking done in {time.perf_counter() - start} seconds.')


# now start benchmark on SLM + LogosDB as RAG
# TODO: Implement this function
def benchmark_slm_rag(df: pd.DataFrame) -> None:
    print('Starting benchmarking on SLM + Logos RAG...')
    start = time.perf_counter()
    res_dir = 'benchmark/results'
    res_file = 'llama_logos.txt'
    res_path = os.path.join(res_dir, res_file)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    topic = df['subject'][0]
    with open(res_path, 'w') as f:
        f.write(f'Topic: {topic}\n')

    for i, row in df.iterrows():
        print(f'Processing row {i}/{len(df)}...')
        question = row['question']
        choices = row['choices']

        suffix_prompt = '''
        You might find the following documents helpful to answer the question.
        If they are irrelevant, just ignore them and use your own reasoning.
        '''

        suffix_2_prompt = '''
        End of documents.
        Remember, only answer with "A", "B", "C", "D", or "E".
        
        Example answer:
        A
        
        Your answer:
        '''

        rag_prompt = PROMPT % (question, choices)
        rag_results = call_rag(query=question)

        # build the prompt for the SLM
        for i, doc in enumerate(rag_results):
            suffix_prompt += f'Document {i+1}:\n{doc}\n'

        # Only add the suffix prompt if there are results from RAG
        if rag_results:
            final_prompt = f'{rag_prompt}\n{suffix_prompt}\n{suffix_2_prompt}'

        res = raw_call(final_prompt, model=OLLAMA_MODEL)
        with open(res_path, 'a') as f:
            f.write(f'{res}\n')
            # f.write(f'Question: {question}\n')
            # f.write(f'Choices: {choices}\n')
            # f.write(f'Answer: {res}\n')
            # f.write('--------------------------\n')

if __name__ == '__main__':
    benchmark_raw(df)
    # benchmark_slm_rag(df)
