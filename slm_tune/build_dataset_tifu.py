from typing import List, Tuple
import time
import json
import pandas as pd

'''
This file is to build dataset for finetuning Qwen2 model. It uses data from Reddit TIFU to build dataset.
'''

def build_dataset(train_file: str, valid_file: str, test_file: str, CHUNK_SIZE: int = 10) -> None:
    # First, load data from json file
    print('Loading Reddit TIFU dataset from json file...')
    json_input = './reddit_tifu/tifu_all_tokenized_and_filtered.json'
    posts = []
    with open(json_input, 'r') as fp:
        for line in fp:
            posts.append(json.loads(line))
    print('Successfully loaded Reddit TIFU dataset')

    # Then, extract only each post's title, selftext, and tldr
    data: List[Tuple[str, str]] = []

    #* tldr is optional
    headers = ['input', 'output']
    for post in posts:
        _input = post['selftext']

        if not post['tldr']:
            _output = post['title']
        else:
            _output = post['tldr'] if len(post['tldr']) >= len(post['title']) else post['title']

        data.append((_input, _output))

    print(f'Merge data through {CHUNK_SIZE}-to-1 mapping...')
    print(f'Original data size: {len(data)}')
    merged_data = []
    for i in range(0, len(data), CHUNK_SIZE):
        # for every data, add dash before it and \n after it
        _input = '\n'.join([f'- {x[0]}' for x in data[i:i+CHUNK_SIZE]])
        _output = '\n'.join([f'- {x[1]}' for x in data[i:i+CHUNK_SIZE]])
        merged_data.append((_input, _output))
    data = merged_data
    print(f'Successfully merged data through {CHUNK_SIZE}-to-1 mapping')
    print(f'Current data size: {len(data)}')
    # train/validation/test split ratio is 70/15/15
    print('Splitting data into train/validation/test...')
    train_size = int(len(data) * 0.7)
    valid_size = int(len(data) * 0.15)

    # Split data
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]
    print('Successfully split data into train/validation/test')

    print('Saving all datasets...')
    df_train = pd.DataFrame(train_data, columns=headers)
    df_valid = pd.DataFrame(valid_data, columns=headers)
    df_test = pd.DataFrame(test_data, columns=headers)
    
    df_train.to_csv(train_file, index=False)
    df_valid.to_csv(valid_file, index=False)
    df_test.to_csv(test_file, index=False)

    print(f"Successfully saved all datasets to {train_file}, {valid_file}, {test_file}")

    
if __name__ == "__main__":
    # 1.2M row / 10 = 120k chunks
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'

    start = time.perf_counter()
    print("Building dataset...")
    
    print(f"Build dataset with train_file: {train_file}, valid_file: {valid_file}, test_file: {test_file}")
    build_dataset(train_file, valid_file, test_file)
    print("Dataset built successfully!!!")
    print(f"Time taken: {time.perf_counter() - start:.2f} seconds")