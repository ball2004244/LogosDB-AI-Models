from train import KeywordExtractionModel
from infer import infer
import time
import pandas as pd
import joblib
import torch

model_path = 'model.pth'
vectorizer_path = 'vectorizer.pkl'
mlb_path = 'mlb.pkl'
dim_file = 'dims.txt'

with open(dim_file, 'r') as file:
    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = map(int, file.read().split()) # assume dim.txt looks like '100 128 10'

model = KeywordExtractionModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load(model_path))

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

vectorizer = joblib.load(vectorizer_path)
mlb = joblib.load(mlb_path)

def test_inference(batch_size=32):
    start = time.perf_counter()
    # Load test data from test.csv
    test_file = 'test.csv'
    test_data = pd.read_csv(test_file)
    raw_texts = test_data['RawText'].tolist()

    expected_keywords = []
    for keywords in test_data['Keywords'].dropna().tolist():
        # Convert expected keywords to a list of lists
        expected_keywords.append(keywords.split(', '))

    num_samples = len(raw_texts)
    all_predicted_keywords = []

    print('Running inference on test data...')
    for start_idx in range(0, num_samples, batch_size):
        print(f'Test on batch {start_idx // batch_size + 1}/{num_samples // batch_size}')

        end_idx = min(start_idx + batch_size, num_samples)
        batch_raw_texts = raw_texts[start_idx:end_idx]

        batch_predicted_keywords = infer(
            model, vectorizer, mlb, batch_raw_texts)
        all_predicted_keywords.extend(batch_predicted_keywords)
        print(f'Finished testing on batch {start_idx // batch_size + 1}/{num_samples // batch_size}')

    print(f'Finished test inference, Time taken: {time.perf_counter() - start:.2f} seconds')

    accuracy = find_accuracy(
        all_predicted_keywords, expected_keywords)
    print(f'Accuracy: {accuracy:.2f} ~ {accuracy * 100:.2f}%')

def find_accuracy(predicted_keywords, expected_keywords):
    total_correct = 0
    total_keywords = 0

    for pred_keywords, exp_keywords in zip(predicted_keywords, expected_keywords):
        pred_set = set(pred_keywords)
        exp_set = set(exp_keywords)

        # Calculate the intersection of predicted and expected keywords
        intersection = pred_set.intersection(exp_set)

        # Count the number of correct predictions
        total_correct += len(intersection)
        # Count the total number of expected keywords
        total_keywords += len(exp_set)

    # Calculate total accuracy
    if total_keywords > 0:
        accuracy = total_correct / total_keywords
    else:
        accuracy = 0

    return accuracy
if __name__ == "__main__":
    BATCH_SIZE = 1024
    test_inference(batch_size=BATCH_SIZE)