from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
# from rich import print
import datetime
import time
import re

'''
Extract keywords from the text with tf-idf vectorization.
'''
def tfidf_summarize(text: str) -> List[str]:
    text_cleaned = re.sub(r"[^\w\s]", "", text).lower()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    vectorizer.fit([text_cleaned])  # Fitting on the single cleaned line
    
    tfidf_matrix = vectorizer.transform([text_cleaned])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[-5:][::-1]
    key_phrases = feature_array[tfidf_sorting]
    return key_phrases

def main():
    LOG_RATE = 10000 # Log every 10k rows
    print('Reading input file...')
    print('Start summarizing with tf-idf...')
    start = time.perf_counter()
    with open('inp.txt', 'r') as file:
        inputs = file.readlines()
        
    with open('out.txt', 'w') as file:
        for i, _input in enumerate(inputs):
            file.write(f'Row {i}: {tfidf_summarize(_input)}\n')

            if i % LOG_RATE != 0:
                continue


            with open('log.txt', 'a') as log_file:
                elapsed = time.perf_counter() - start
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f'[{current_time}] Elapsed: {elapsed:.2f} sec ({elapsed/3600:.2f} hours) - Processed {i} rows\n')
    
    # Expect 0.342h for 1M rows -> 2M ~ 0.684h ~ 41min
    # -> 10M ~ 3.42h, 100M ~ 34.2h, 1B ~ 342h
    cur = time.perf_counter() - start
    print('Finished summarizing with tf-idf')
    print(f'Time taken: {cur:.2f} seconds (or {cur/3600:.2f} hours)')
    print(f'Average: {cur/len(inputs):.2f} sec/row')
    print('Output saved to out.txt')
    
    with open('stats.txt', 'w') as file:
        file.write(f'Time taken: {cur:.6f} seconds (or {cur/3600:.6f} hours)\n')
        file.write(f'Average: {cur/len(inputs):.6f} sec/row\n')

if __name__ == "__main__":
    main()