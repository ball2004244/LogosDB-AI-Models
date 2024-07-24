
'''
This file capable of preprocessing the data and split it into train, validation, and test sets.
'''

from typing import Tuple
from tfidf_summary import tfidf_summarize
import multiprocessing as mp
import time

import pandas as pd
import numpy as np

def tfidf_summarize_helper(text: str) -> str:
    return ', '.join(tfidf_summarize(text))

def build_train_data(filename: str) -> pd.DataFrame:
    '''
    Build data with TF-IDF vectorizer
    '''
    
    df = pd.read_csv(filename)[['Question', 'Answer']]
    # first concat the Question and Answer columns to form a new column named 'RawText'
    df['RawText'] = df['Question'] + ' ' + df['Answer']
    
    # now drop all empty str or NaN values
    for col in df.columns:
        df = df[df[col].notna()]
        df = df[df[col] != '']
    df.reset_index(drop=True, inplace=True)
    
    # Use multiprocessing to apply tfidf_summarize
    with mp.Pool() as pool:
        df['Keywords'] = pool.map(tfidf_summarize_helper, df['RawText'])
        
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # save the data to a new file
    df.to_csv(filename, index=False)

    return df

def split_data(df: pd.DataFrame, train_file: str, valid_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split the data into train, validation, and test sets
    '''
    ratio = {
        'train': 0.7,
        'valid': 0.15,
        'test': 0.15
    }

    train = df.iloc[:int(ratio['train'] * len(df))].dropna()
    valid = df.iloc[int(ratio['train'] * len(df)):
                    int((ratio['train'] + ratio['valid']) * len(df))].dropna()
    test = df.iloc[int((ratio['train'] + ratio['valid']) * len(df)):].dropna()
    
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    test.to_csv(test_file, index=False)

    return train, valid, test
    
def main() -> None:
    raw_file = 'single_qna.csv'
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'

    start = time.perf_counter()
    print('Preprocessing data with multiprocessing...')
    df = build_train_data(raw_file)
    print(f'Preprocessing done!, Time taken: {time.perf_counter() - start:.2f} seconds')

    print('Preparing train, validation, and test sets...')
    t_start = time.perf_counter()
    split_data(df, train_file, valid_file, test_file)
    print(f'Data split done!, Time taken: {time.perf_counter() - t_start:.2f} seconds')

if __name__ == '__main__':
    main()
