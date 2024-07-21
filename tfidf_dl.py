import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
'''
Train a deep learning model to reconstruct the TF-IDF vectors.
This model can process multiple documents at once, instead of one by one.
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

# Example text data and labels
print('Reading input file...')
start = time.perf_counter()
#* Read from a text file
# texts = []
# with open('temp.txt', 'r') as file:
#     texts = file.readlines()

#* Read from a CSV file
df = pd.read_csv('single_qna.csv')[['Question', 'Answer']]
df.fillna('', inplace=True)
texts = df['Question'] + ' ' + df['Answer']
print(f'Reading input file took {time.perf_counter() - start:.2f} seconds')


# Generate TF-IDF vectors
print('Generating TF-IDF vectors...')
start = time.perf_counter()
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X = vectorizer.fit_transform(texts).toarray()
print(f'Generating TF-IDF vectors took {time.perf_counter() - start:.2f} seconds')

# Convert arrays to PyTorch tensors and move to the specified device
print('Converting to PyTorch tensors & moving to device...')
start = time.perf_counter()
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
print(f'Conversion took {time.perf_counter() - start:.2f} seconds')
class AdvancedTFIDFModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedTFIDFModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Reduced from 1024 to 512
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)  # Reduced from 512 to 256
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model(model: nn.Module, X: torch.Tensor, epochs: int, batch_size: int = 32) -> nn.Module:
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(X, X)  # Using X as both input and target since it's an autoencoder
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print('Start Training...')
    start = time.perf_counter()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    print(f'Time taken: {time.perf_counter() - start:.2f} seconds')
    print('Finished Training')
    return model

# Take roughly 1252s ~ 20 minutes for this config
model = AdvancedTFIDFModel(input_dim=X.shape[1], output_dim=X.shape[1]).to(device)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 100
BATCH_SIZE = 64

model = train_model(model, X_tensor, EPOCHS, BATCH_SIZE)

# Save the model
model_file = 'tfidf_model.pth'
torch.save(model.state_dict(), model_file)
print(f'Model saved to {model_file}')

# Inference
print('Starting Inference...')
start = time.perf_counter()
with open('temp.txt', 'r') as file:
    inf_texts = file.readlines()

inf_X = vectorizer.transform(inf_texts).toarray()
inf_X_tensor = torch.tensor(inf_X, dtype=torch.float32).to(device)

feature_names = vectorizer.get_feature_names_out()
with torch.no_grad():
    inf_reconstructed = model(inf_X_tensor)
    inf_thresholded_outputs = torch.where(inf_reconstructed > 0.1, inf_reconstructed, torch.tensor(0.0).to(device))
    inf_top_keywords_per_doc = torch.topk(inf_thresholded_outputs, 5).indices.cpu().numpy()

inf_top_keywords = feature_names[inf_top_keywords_per_doc]
print("Top 5 keywords per document:")
for i, keywords in enumerate(inf_top_keywords):
    print(f"Document {i+1}: {keywords}")

print(f'Inference took {time.perf_counter() - start:.2f} seconds')