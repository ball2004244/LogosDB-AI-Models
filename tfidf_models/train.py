import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
# Train a new DeepLearning model based on TF-IDF modelling.

# First, generate train data using TF-IDF vectorizer
# train.csv format: RawText, Keywords
# e.g: 
# RawText: "Wormholes, hypothetical topological features of spacetime, have long captivated the imagination of scientists and science fiction enthusiasts alike. These theoretical tunnels through space and time offer the tantalizing possibility of shortcuts across the universe, potentially allowing for faster-than-light travel. First conceptualized in 1935 by Einstein and Rosen, wormholes emerge from solutions to the equations of general relativity. While mathematically possible, the existence of traversable wormholes faces significant challenges. They would require exotic matter with negative energy density to remain open and stable, a concept that pushes the boundaries of known physics. If they exist, wormholes could connect distant regions of space-time, even linking different universes or timelines. This property has led to speculation about their potential for time travel, though the paradoxes this might create remain unresolved. Despite their theoretical intrigue, no observational evidence for wormholes has been found. Current research focuses on refining mathematical models and exploring potential detection methods. As our understanding of quantum gravity and the nature of spacetime evolves, wormholes continue to serve as a fascinating intersection of theoretical physics, cosmology, and our quest to unravel the universe's deepest mysteries.",
# Keywords: "wormholes, spacetime, theoretical physics, general relativity

# Then, train the model using the generated data
# The model must be able to handle multiple inputs and give multiple outputs instead of single input and single output of original TF-IDF model

# Finally, test the model using test data
# test.csv format: RawText, Keywords


'''
This file is used to train a keyword extraction model using an LSTM neural network
'''

def preprocess_data(df, vectorizer=None, mlb=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['RawText'])
    else:
        X = vectorizer.transform(df['RawText'])
    
    if mlb is None:
        mlb = MultiLabelBinarizer(sparse_output=True)
        y = mlb.fit_transform(df['Keywords'].apply(lambda x: x.split(', ') if isinstance(x, str) else []))
    else:
        y = mlb.transform(df['Keywords'].apply(lambda x: x.split(', ') if isinstance(x, str) else []))
    
    return X, y, vectorizer, mlb

class KeywordExtractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KeywordExtractionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation

    def forward(self, x):
        x = x.unsqueeze(1)  # Add an extra dimension: (batch_size, 1, input_dim)
        h_lstm, _ = self.lstm(x)
        h_lstm = h_lstm[:, -1, :]  # Take the last hidden state
        out = self.fc(h_lstm)
        out = self.sigmoid(out)  # Apply sigmoid activation
        return out

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return self.texts.shape[0]  # Use shape[0] to get the number of samples

    def __getitem__(self, idx):
        text = self.texts[idx].toarray().squeeze()
        label = self.labels[idx].toarray().squeeze()
        return torch.tensor(text, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def start_training():
    BATCH_SIZE = 1024
    EPOCHS = 100
    CHUNK_SIZE = 100000
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'
    model_file = 'model.pth'

    print('CONFIGURATION:')
    print(f'BATCH_SIZE: {BATCH_SIZE}\nEPOCHS: {EPOCHS}\nCHUNK_SIZE: {CHUNK_SIZE}')
    start = time.perf_counter()
    print('Reading training data...')
    df_iter = pd.read_csv(train_file, chunksize=CHUNK_SIZE)
    
    vectorizer = None
    mlb = None
    model = None
    criterion = nn.BCELoss()
    optimizer = None

    total_chunks = sum(1 for _ in pd.read_csv(train_file, chunksize=CHUNK_SIZE))

    for i, chunk in enumerate(df_iter):
        print(f'Processing chunk {i + 1}/{total_chunks}...')
        
        print('Preprocessing data...')
        X, y, vectorizer, mlb = preprocess_data(chunk, vectorizer, mlb)
        
        # save all dimensions, mlb and vectorizer to a file
        dimfile = 'dims.txt'
        vectorizer_file = 'vectorizer.pkl'
        mlb_file = 'mlb.pkl'
        with open(dimfile, 'w') as file:
            file.write(f'{X.shape[1]} {128} {y.shape[1]}')
        joblib.dump(vectorizer, vectorizer_file)
        joblib.dump(mlb, mlb_file)
        
        
        print('Creating dataset and dataloader...')
        train_dataset = TextDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        input_dim = X.shape[1]
        hidden_dim = 128
        output_dim = y.shape[1]
        
        if model is None:
            print(f'Initializing model with Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}')
            model = KeywordExtractionModel(input_dim, hidden_dim, output_dim)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print('Training model...')
        train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCHS)
        print(f'Finished training chunk {i + 1}/{total_chunks}')

    print('Saving final model...')
    torch.save(model.state_dict(), model_file)
    print(f'Final model saved to {model_file}')
    print(f'Training completed!, Time taken: {time.perf_counter() - start:.2f} seconds')
    
    # Now we can use the trained model for validate and test
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)
    
    # Preprocess validation and test data
    X_valid, y_valid, _, _ = preprocess_data(valid_data, vectorizer, mlb)
    X_test, y_test, _, _ = preprocess_data(test_data, vectorizer, mlb)
    
    valid_dataset = TextDataset(X_valid, y_valid)
    test_dataset = TextDataset(X_test, y_test)
    
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Evaluate the model on validation and test data
    model.eval()
    
    valid_predictions = make_predictions(model, valid_loader)
    test_predictions = make_predictions(model, test_loader)
    
    valid_accuracy = find_accuracy(valid_predictions, valid_data['Keywords'])
    test_accuracy = find_accuracy(test_predictions, test_data['Keywords'])
    
    print(f'Validation accuracy: {valid_accuracy:.2f} ~ {valid_accuracy * 100:.2f}%')
    print(f'Test accuracy: {test_accuracy:.2f} ~ {test_accuracy * 100:.2f}%')
    
# Define the prediction function
def make_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data['RawText']  # Adjust this line based on your dataset structure
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())
    return predictions

# Define the accuracy function
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

if __name__ == '__main__':
    start_training()
    