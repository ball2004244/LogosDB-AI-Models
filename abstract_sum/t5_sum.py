import time
import os
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

'''
This script demonstrates how to use T5 to do abstractive summarization on a batch of documents.
'''
# pip install transformers torch

class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = T5Tokenizer.from_pretrained('T5-base')

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        inputs = self.tokenizer.encode("summarize: " + document, return_tensors='pt', max_length=512, truncation=True)
        return inputs.squeeze(0)

def inference_batch(model, batch):
    inputs = batch.to('cuda')
    output = model.generate(inputs, min_length=50, max_length=100)
    summaries = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return summaries

def mass_abstract_sum(docs: List[str]) -> List[str]:
    summaries = []
    dataset = DocumentDataset(docs)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=os.cpu_count())
    
    model_name = 'T5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True).to('cuda')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    for batch in dataloader:
        summaries.extend(inference_batch(model, batch))
        
    return summaries

if __name__ == "__main__":
    raw_document = """
    Nature is a tapestry of life, woven with vibrant hues and textures. From towering mountains to tranquil oceans, each element plays a vital role in this intricate ecosystem. The rustling leaves whisper secrets, the gurgling streams sing melodies, and the sun's warm embrace nourishes all. Nature offers solace to weary souls, inspiring awe and wonder. Its delicate balance reminds us of our interconnectedness and the importance of preserving this precious gift for generations to come.
    """
    
    docs = [raw_document] * 1000
    dataset = DocumentDataset(docs)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=os.cpu_count())

    model_name = 'T5-base'
    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True).to('cuda')
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    summaries = []
    start = time.perf_counter()
    for batch in dataloader:
        summaries.extend(inference_batch(model, batch))

    print(f'Time taken: {time.perf_counter() - start:.2f} seconds')
    with open('summary.txt', 'w') as f:
        for summary in summaries:
            f.write(summary + '\n')
    print(summaries[:5])