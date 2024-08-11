from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import time
import nltk
import networkx as nx
import numpy as np
import multiprocessing as mp

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def store_mp_config(config_file: str='mp_config.txt') -> None:
    num_cores = mp.cpu_count()
    start_method = mp.get_start_method()

    with open(config_file, 'w') as file:
        file.write(f'Number of CPU cores: {num_cores}\n')
        file.write(f'Multiprocessing start method: {start_method}\n')


def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def build_similarity_matrix(sentences):
    stop_words = list(set(stopwords.words('english')))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    n_components = min(100, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=7)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    similarity_matrix = np.dot(reduced_tfidf_matrix, reduced_tfidf_matrix.T)
    return similarity_matrix

def rank_sentences(similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
    return scores

def extract_summary(sentences, scores, num_sentences=3):
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join(
        sentence for _, sentence in ranked_sentences[:num_sentences])
    return summary

def process_text(text):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    scores = rank_sentences(similarity_matrix)
    summary = extract_summary(sentences, scores)
    return summary

def mass_extract_summaries(inputs: List[str]) -> List[str]:
    print(f'Start summarize on multiprocessing mode with {mp.cpu_count()} cores') 
    with mp.Pool() as pool:
        summaries = pool.map(process_text, inputs)
    return summaries

def compare() -> None:
    store_mp_config()
    print('Starting reading input')
    input_file = 'data/inp_2M.txt'
    output_file = 'summary_output.txt'

    with open(input_file, 'r') as file:
        user_inputs = file.readlines()

    num_data = [200000, 500000, 1000000, 2000000]
    time_file = 'time_vs_rows_mp.txt'

    print('Running extractive summarization on different number of rows')
    print(f'All rows to be processed: {num_data}')
    with open(time_file, 'w') as time_file:
        for n in num_data:
            print(f'Processing {n} rows')
            sliced_inputs = user_inputs[:n]
            
            start_time = time.perf_counter()
            summaries = mass_extract_summaries(sliced_inputs)
            elapsed_time = time.perf_counter() - start_time
            
            time_file.write(f'{n} {elapsed_time:.2f}\n')
            print(f'Completed {n} rows in {elapsed_time:.2f} seconds')

    print('All done!')

if __name__ == '__main__':
    compare()
