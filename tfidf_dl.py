import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
'''
Train a deep learning model to reconstruct the TF-IDF vectors.
This model can process multiple documents at once, instead of one by one.
'''
# Example text data and labels
texts = []
with open('temp.txt', 'r') as file:
    texts = file.readlines()

# Generate TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X = vectorizer.fit_transform(texts).toarray()

# Convert arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# Define a more complex PyTorch model
class AdvancedTFIDFModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdvancedTFIDFModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No activation, aiming for reconstruction
        return x

model = AdvancedTFIDFModel(input_dim=X.shape[1], output_dim=X.shape[1])

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):  # Increased epochs
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, X_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/20], Loss: {loss.item():.4f}')

# Post-processing to extract top 5 keywords per document
with torch.no_grad():
    reconstructed = model(X_tensor)
    # Apply a threshold to filter out low values
    thresholded_outputs = torch.where(reconstructed > 0.1, reconstructed, torch.tensor(0.0))
    top_keywords_per_doc = torch.topk(thresholded_outputs, 5).indices.numpy()

# Convert indices to words
feature_names = np.array(vectorizer.get_feature_names_out())
top_keywords = feature_names[top_keywords_per_doc]

print("Top 5 keywords per document:")
print(top_keywords)