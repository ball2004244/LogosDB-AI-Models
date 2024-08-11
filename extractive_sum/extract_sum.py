from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import time
import nltk
import networkx as nx
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

'''
This file contains a ML algorithm that extracts summaries from a given list of texts.
The algorithm uses the extractive summarization technique to rank the sentences in the text
and extract the most important sentences to form a summary.
'''

# Preprocess the text by tokenizing it into sentences


def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

# Build a similarity matrix based on the sentences


def build_similarity_matrix(sentences):
    stop_words = list(set(stopwords.words('english')))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    n_components = min(100, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=7)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    similarity_matrix = np.dot(reduced_tfidf_matrix, reduced_tfidf_matrix.T)
    return similarity_matrix

# Rank the sentences based on the similarity matrix


def rank_sentences(similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
    return scores

# Extract the summary from the ranked sentences


def extract_summary(sentences, scores, num_sentences=3):
    ranked_sentences = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join(
        sentence for _, sentence in ranked_sentences[:num_sentences])
    return summary

# Process the text by preprocessing, building similarity matrix, ranking sentences, and extracting summary


def process_text(text):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    scores = rank_sentences(similarity_matrix)
    summary = extract_summary(sentences, scores)
    return summary


'''
Mass extract summaries from a list of texts and write the summaries to a list of summaries.
Input: List of raw texts
Output: List of summarized texts
'''


def mass_extract_summaries(inputs: List[str]) -> List[str]:
    return [process_text(text) for text in inputs]


'''
This function is used to run the extractive summarization algorithm on different number of rows,
to find relationship between the number of rows and the time taken to process them.
'''


def compare() -> None:
    print('Starting reading input')
    start_time = time.perf_counter()

    input_file = 'summary_input.txt'
    output_file = 'summary_output.txt'

    with open(input_file, 'r') as file:
        user_inputs = file.readlines()

    num_data = [25000, 50000, 75000, 100000, 125000]
    time_file = 'time_vs_rows.txt'

    print('Running extractive summarization on different number of rows')
    print(f'All rows to be processed: {num_data}')
    with open(time_file, 'w') as time_file:
        for n in num_data:
            print(f'Processing {n} rows')
            sliced_inputs = user_inputs[:n]
            summaries = mass_extract_summaries(sliced_inputs)

            elapsed_time = time.perf_counter() - start_time
            time_file.write(f'{n} {elapsed_time:.2f}\n')
            print(f'Completed {n} rows in {elapsed_time:.2f} seconds')

    print('All done!')


if __name__ == '__main__':
    # input_file = 'summary_input.txt'
    # output_file = 'summary_output.txt'
    # mass_extract_summaries(input_file, output_file)
    compare()
