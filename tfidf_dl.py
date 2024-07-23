# Train a new DeepLearning model based on TF-IDF modelling.

# First, generate train data using TF-IDF vectorizer
# train.csv format: RawText, Keywords
# e.g: 
# RawText: "Wormholes, hypothetical topological features of spacetime, have long captivated the imagination of scientists and science fiction enthusiasts alike. These theoretical tunnels through space and time offer the tantalizing possibility of shortcuts across the universe, potentially allowing for faster-than-light travel. First conceptualized in 1935 by Einstein and Rosen, wormholes emerge from solutions to the equations of general relativity. While mathematically possible, the existence of traversable wormholes faces significant challenges. They would require exotic matter with negative energy density to remain open and stable, a concept that pushes the boundaries of known physics. If they exist, wormholes could connect distant regions of space-time, even linking different universes or timelines. This property has led to speculation about their potential for time travel, though the paradoxes this might create remain unresolved. Despite their theoretical intrigue, no observational evidence for wormholes has been found. Current research focuses on refining mathematical models and exploring potential detection methods. As our understanding of quantum gravity and the nature of spacetime evolves, wormholes continue to serve as a fascinating intersection of theoretical physics, cosmology, and our quest to unravel the universe's deepest mysteries.",
# Keywords: "wormholes, spacetime, theoretical physics, general relativity

# Then, train the model using the generated data

# Finally, save the model to a file

from typing import Tuple
from tfidf_summary import tfidf_summarize
import multiprocessing as mp
import functools
import time

import pandas as pd
import numpy as np

# Read the raw data
def read_raw_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

def tfidf_summarize_helper(text: str) -> str:
    return ', '.join(tfidf_summarize(text))

def build_train_data(df: pd.DataFrame, filename: str) -> None:
    '''
    Build data with TF-IDF vectorizer
    '''
    
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

    return

def split_data(df: pd.DataFrame, train_file: str, valid_file: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split the data into train, validation, and test sets
    '''
    ratio = {
        'train': 0.7,
        'valid': 0.15,
        'test': 0.15
    }
    
    train, valid, test = np.split(
        df.sample(frac=1, random_state=42), 
        [int(ratio['train'] * len(df)), 
         int((ratio['train'] + ratio['valid']) * len(df))]
    )
    
    train.to_csv(train_file, index=False)
    valid.to_csv(valid_file, index=False)
    test.to_csv(test_file, index=False)

    return train, valid, test

class TFIDFDeepLearning:
    def __init__(self):
        pass

    def train(self, train_file: str) -> None:
        pass

    def predict(self, text: str) -> str:
        pass

    

def main() -> None:
    raw_file = 'single_qna.csv'
    processed_file = 'processed_qna.csv'
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'

    start = time.perf_counter()
    print('Preprocessing data with multiprocessing...')
    df = read_raw_data(raw_file)[['Question', 'Answer']]
    build_train_data(df, processed_file)
    print('Preprocessing done!, Time taken:', time.perf_counter() - start)

    print('Preparing train, validation, and test sets...')
    t_start = time.perf_counter()
    df = read_raw_data(processed_file)
    split_data(df, train_file, valid_file, test_file)
    print('Data split done!, Time taken:', time.perf_counter() - t_start)

    print('Start training...')
    t_start = time.perf_counter()
    
    print('Finished training!, Time taken:', time.perf_counter() - t_start)
    return

if __name__ == '__main__':
    main()
