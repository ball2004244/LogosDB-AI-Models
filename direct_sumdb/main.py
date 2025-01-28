from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

'''
This code implement SumDB from Scratch instead of relying on Margo Services
'''
class SumDB:
    def __init__(self) -> None:
        # Load E5 model
        self.model = SentenceTransformer('intfloat/e5-base-v2')
        # Store embeddings and metadata
        self.embeddings = []
        self.metadata = []
        
    def insert(self, vectors: List[Dict[str, str]], CHUNK_SIZE: int = 128) -> bool:
        '''
        Insert vectors into SumDB in chunks
        '''
        try:
            for i in range(0, len(vectors), CHUNK_SIZE):
                batch = vectors[i:i + CHUNK_SIZE]
                # Generate embeddings for summaries
                texts = [f"passage: {doc['summary']}" for doc in batch]
                embeddings = self.model.encode(texts, convert_to_tensor=True)
                
                # Store embeddings and metadata
                self.embeddings.append(embeddings)
                self.metadata.extend(batch)
                
            # Concatenate all embeddings
            if len(self.embeddings) > 0:
                self.embeddings = torch.cat(self.embeddings, dim=0)
            
            return True
            
        except Exception as e:
            print(f'Error at SumDB insert: {e}')
            return False

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, str]]:
        '''
        Query similar content using cosine similarity
        '''
        try:
            # Generate query embedding
            query_text = f"query: {query_text}"
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)

            # Calculate similarities
            similarities = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                self.embeddings.cpu().numpy()
            )[0]

            # Get top-k results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                result = self.metadata[idx].copy()
                result['_score'] = float(similarities[idx])
                results.append(result)
                
            return results
            
        except Exception as e:
            print(f'Error at SumDB query: {e}')
            return []

    def delete_all(self) -> bool:
        '''
        Delete all vectors
        '''
        try:
            self.embeddings = []
            self.metadata = []
            return True
            
        except Exception as e:
            print(f'Error at SumDB delete_all: {e}')
            return False
        
if __name__ == '__main__':
    # Test SumDB
    db = SumDB()
    vectors = [
        {'summary': 'This is a test summary 1'},
        {'summary': 'This is a test summary 2'},
        {'summary': 'This is a test summary 3'},
    ]
    db.insert(vectors)
    results = db.query('This is a test query', top_k=2)
    print(results)
    db.delete_all()
    results = db.query('This is a test query', top_k=2)
    print(results)