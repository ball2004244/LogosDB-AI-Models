from collections import defaultdict
from rich import print
from keybert import KeyBERT
import spacy
import time

'''
This class provides a simple text summarization functionality using the KeyBERT library.
It returns the top keywords and sentences based on the weighted presence of keywords (which is a brief summary).

pip install keybert
'''

class TextSummarizer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('sentencizer')
        self.kw_model = KeyBERT()

    def summarize(self, input_text: int, summary_percent=0.2, verbose=False) -> tuple:
        start = time.perf_counter()

        # Extract keywords with their scores
        raw_keywords = self.kw_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 1), top_n=10)
        keyword_scores = {kw[0]: kw[1] for kw in raw_keywords}

        # Tokenize the text into sentences
        doc = self.nlp(input_text)
        sentences = [sent.text for sent in doc.sents]

        # Score sentences based on the weighted presence of keywords
        sentence_scores = defaultdict(int)
        for sentence in sentences:
            sentence = sentence.lower()
            for keyword, score in keyword_scores.items():
                sentence_scores[sentence] += sentence.count(keyword) * score

        # Select top sentences based on their scores
        num_sentences = max(int(len(sentences) * summary_percent), 1)  # Ensure at least one sentence is selected
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

        top_keywords = list(keyword_scores.keys())

        # Output
        if verbose:
            print('Input Text:', input_text)
            print('Keywords:', top_keywords)
            print('Top Sentences:', top_sentences)
            print('Time taken:', (time.perf_counter() - start), 'seconds')

        return top_keywords, top_sentences

    def mass_summarize(self, input_texts: list, summary_percent=0.2) -> list:
        summaries = []
        for text in input_texts:
            summaries.append(self.summarize(text, summary_percent))
        return summaries

# Example usage
if __name__ == "__main__":
    summarizer = TextSummarizer()
    input_file = 'inp.txt'
    with open(input_file, 'r') as file:
        input_text = file.read()
    print('Input Text:', input_text)
    print('Summary:', summarizer.summarize(input_text))
