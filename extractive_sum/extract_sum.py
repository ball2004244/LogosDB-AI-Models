import time
import itertools
import nltk
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = ' '.join(sentence for _, sentence in ranked_sentences[:num_sentences])
    return summary

def process_text(text):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    scores = rank_sentences(similarity_matrix)
    summary = extract_summary(sentences, scores)
    return summary

def main():
    print('Starting reading input')
    start_time = time.perf_counter()
    
    input_file = 'summary_input.txt'
    output_file = 'summary_output.txt'
    
    with open(input_file, 'r') as file:
        user_inputs = file.readlines()

    num_data = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    time_file = 'time_vs_rows.txt'
    
    print('Running extractive summarization on different number of rows')
    with open(time_file, 'w') as time_file:
        with open(output_file, 'w') as summary_file:
            for n in num_data:
                print(f'Processing {n} rows')
                sliced_inputs = user_inputs[:n]
                summaries = [process_text(text) for text in sliced_inputs]
                summary_file.write('\n'.join(summaries) + '\n')
                
                elapsed_time = time.perf_counter() - start_time
                time_file.write(f'{n} {elapsed_time:.2f}\n')
                print(f'Completed {n} rows in {elapsed_time:.2f} seconds')
    
    print('All done!')

if __name__ == '__main__':
    main()