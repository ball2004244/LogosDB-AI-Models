# Define NPY_NO_DEPRECATED_API to disable deprecated NumPy API
# This must be done before any NumPy headers are included
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import nltk
import networkx as nx
import multiprocessing as mp
import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, free

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# A multi-processing version of the extractive summarization algorithm.

def count_words(text):
    return len(text.split())

cpdef list preprocess_text(str text):
    cdef list sentences = sent_tokenize(text)
    # Filter out empty sentences
    filtered_sentences = []
    for s in sentences:
        if s.strip():
            filtered_sentences.append(s)
    return filtered_sentences

cpdef cnp.ndarray build_similarity_matrix(list sentences):
    vectorizer = TfidfVectorizer()  # Do not remove stopwords
    tfidf_matrix = vectorizer.fit_transform(sentences).toarray()

    # Let this be None for now
    if tfidf_matrix.shape[1] < 2:
        return None

    cdef int n_components = min(tfidf_matrix.shape[1] - 1, 100)  # Subtract 1 from tfidf_matrix.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=7)
        reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

    # Calculate similarity matrix directly from the reduced TF-IDF matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] similarity_matrix = np.dot(reduced_tfidf_matrix, reduced_tfidf_matrix.T)
    return similarity_matrix

cpdef dict rank_sentences(cnp.ndarray similarity_matrix):
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-6)
    return scores

cpdef str extract_summary(list sentences, dict scores, int num_sentences=3):
    cdef list ranked_sentences = []
    cdef int i
    cdef str s
    for i in range(len(sentences)):
        ranked_sentences.append((scores[i], sentences[i]))
    ranked_sentences.sort(reverse=True)
    cdef str summary = ''
    for _, sentence in ranked_sentences[:num_sentences]:
        summary += sentence + ' '
    return summary.strip()

cpdef str process_text(str text):
    cdef list sentences = preprocess_text(text)
    if not sentences:
        print("No sentences found after preprocessing.")
        return ''
    
    cdef int word_count = count_words(text)
    filtered_sentences = []
    for sentence in sentences:
        if word_count < 100:
            filtered_sentences.append(sentence)
        else:
            filtered_sentences.append(sentence)

    if not filtered_sentences:
        print("No filtered sentences found.")
        return ''

    cdef cnp.ndarray similarity_matrix = build_similarity_matrix(filtered_sentences)

    if similarity_matrix is None:
        print("Similarity matrix is None.")
        return ''
    cdef dict scores = rank_sentences(similarity_matrix)
    cdef str summary = extract_summary(filtered_sentences, scores, num_sentences=3)  # Ensure top 3 sentences are returned
    return summary

cpdef list mass_extract_summaries(list inputs):
    print(f'Start summarize on multiprocessing mode with {mp.cpu_count()} cores')
    summaries = []
    for input_text in inputs:
        # print(f"Processing input: {input_text[:100]}...")  # Print the first 100 characters of the input for debugging
        summary = process_text(input_text)
        if summary == '':
            print("Failed to generate summary.")
        summaries.append(summary)
    return summaries