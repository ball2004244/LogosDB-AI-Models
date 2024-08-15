from typing import List
from train import KeywordExtractionModel
import torch
import joblib

'''
This file is used to perform inference using the trained model
'''


def infer(model, vectorizer, mlb, raw_texts: List[str]) -> List[List[str]]:
    # Determine the device from the model parameters
    device = next(model.parameters()).device

    # Preprocess the raw texts using the same vectorizer used during training
    X = vectorizer.transform(raw_texts).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)  # Move tensor to the appropriate device

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(X_tensor)

    # Convert the output probabilities to binary predictions (e.g., using a threshold of 0.5)
    predictions = (outputs > 0.5).int()

    # Convert the binary predictions back to keyword labels
    predicted_keywords = mlb.inverse_transform(predictions.cpu().numpy())  # Move predictions back to CPU

    return predicted_keywords


def run_inference(raw_texts: List[str]) -> List[List[str]]:
    # Load the trained model, vectorizer, and mlb (assuming they are saved)
    model_path = 'model.pth'
    vectorizer_path = 'vectorizer.pkl'
    mlb_path = 'mlb.pkl'
    dim_file = 'dims.txt'

    with open(dim_file, 'r') as file:
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = map(int, file.read().split()) # assume dim.txt looks like '100 128 10'

    model = KeywordExtractionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(model_path))

    # Load the saved vectorizer and MultiLabelBinarizer
    vectorizer = joblib.load(vectorizer_path)
    mlb = joblib.load(mlb_path)

    predicted_keywords = infer(model, vectorizer, mlb, raw_texts)

    return predicted_keywords

if __name__ == "__main__":
    # Example usage with a list of strings
    example_texts = [
        "Wormholes, hypothetical topological features of spacetime, have long captivated the imagination of scientists and science fiction enthusiasts alike.",
        "Quantum computing leverages the principles of quantum mechanics to perform computations at speeds unimaginable with classical computers.",
        "Artificial intelligence and machine learning are transforming industries by enabling systems to learn from data and make decisions."
    ]
    print(run_inference(example_texts))