'''
This file will compared generated answers from SLM with given answers in the dataset to measure the accuracy of the model.
'''
import os
from datasets import load_dataset
import pandas as pd

ds = load_dataset("cais/mmlu", "college_computer_science")
df = pd.DataFrame(ds['test'])

ANSWER_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}
def measure_raw(df: pd.DataFrame) -> None:
    print('Starting measuring on raw SLM...')
    res_dir = 'benchmark/results'
    res_file = 'llama_raw.txt'
    res_path = os.path.join(res_dir, res_file)
    with open(res_path, 'r') as f:
        # Skip the first row (topic)
        f.readline()
        res = f.readlines()
    correct = 0
    wrong = 0
    for i, row in df.iterrows():
        print(f'Processing row {i}/{len(df)}...')
        raw_answer = res[i-1].strip()
        answer = row['answer']
        if ANSWER_MAP[raw_answer] == answer:
            correct += 1
        else:
            wrong += 1
            
    # save the result to a file
    result_file = 'llama_raw_stats.txt'
    result_path = os.path.join(res_dir, result_file)
    with open(result_path, 'w') as f:
        f.write(f'Correct: {correct}, Wrong: {wrong}, Accuracy: {correct/len(df)}')
    print(f'Save result to {result_path}')
if __name__ == '__main__':
    measure_raw(df)