# Save this file as extract_sum_mp.pyx

# Disable deprecated NumPy API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
import networkx as nx
import numpy as np
import multiprocessing as mp

cimport numpy as cnp

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

cpdef list preprocess_text(str text):
    cdef list sentences = sent_tokenize(text)
    return sentences

cpdef cnp.ndarray build_similarity_matrix(list sentences):
    cdef list stop_words = list(set(stopwords.words('english')))
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()

    if tfidf_matrix.shape[1] == 0:
        raise ValueError("TF-IDF matrix has zero columns. Check the input data.")

    n_components = min(100, tfidf_matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=7)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    if np.isnan(reduced_tfidf_matrix).any():
        raise ValueError("NaN values encountered in reduced TF-IDF matrix.")

    similarity_matrix = np.dot(reduced_tfidf_matrix, reduced_tfidf_matrix.T)
    return similarity_matrix

cpdef dict rank_sentences(cnp.ndarray similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
    return scores

cpdef str extract_summary(list sentences, dict scores, int num_sentences=3):
    ranked_sentences = sorted(
        [(scores[i], s) for i, s in enumerate(sentences)], reverse=True)
    summary = ' '.join(
        [sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

cpdef str process_text(str text):
    sentences = preprocess_text(text)
    similarity_matrix = build_similarity_matrix(sentences)
    scores = rank_sentences(similarity_matrix)
    summary = extract_summary(sentences, scores)
    return summary

cpdef list mass_extract_summaries(list inputs):
    print(f'Start summarize on multiprocessing mode with {mp.cpu_count()} cores')
    with mp.Pool() as pool:
        summaries = pool.map(process_text, inputs)
    return summaries