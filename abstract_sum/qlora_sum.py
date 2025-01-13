import os
import time
import torch
import warnings
import pandas as pd
from datasets import load_dataset
from typing import List
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# This is the main code of abstract sum with QLora
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

'''
Do summarize with QLora fine-tune flan T5-base
'''

# pip install transformers torch peft datasets pandas

# Define the DocumentDataset class
class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        document = self.documents[idx]
        inputs = self.tokenizer.encode(
            "summarize to 4 sentences: " + document, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        return inputs.squeeze(0)

# Define the inference_batch function
def inference_batch(model, tokenizer, batch):
    inputs = batch.to('cuda')
    # Extract the underlying model
    base_model = model.base_model
    output = base_model.generate(inputs, max_new_tokens=512, num_beams=3, do_sample=True, temperature=0.7)
    summaries = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in output]
    return summaries

# Define the mass_abstract_sum function
def mass_qlora_abstract_sum(docs: List[str]) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    summaries = []
    dataset = DocumentDataset(docs)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Load the base model and the fine-tuned model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = PeftModel.from_pretrained(base_model, "RMWeerasinghe/flan-t5-base-finetuned-QLoRA-v2").to(device)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    for batch in dataloader:
        summaries.extend(inference_batch(model, tokenizer, batch))

    return summaries