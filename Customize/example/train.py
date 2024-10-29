# This script is to train the model
# Generate data
from example import CircleClassifier, generate_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split


def generate_data(num_samples=1000):
    X = np.random.uniform(-1.5, 1.5, (num_samples, 2))  # Random points in the square [-1.5, 1.5] x [-1.5, 1.5]
    y = np.array([1 if x**2 + y**2 <= 1 else 0 for x, y in X])  # Label 1 if inside the circle, else 0
    return X, y

# Prepare data
X, y = generate_data(2000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Instantiate the model, define loss and optimizer
model = CircleClassifier()
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on test data
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).float().mean()
    print(f'Test Accuracy: {accuracy.item() * 100:.2f}%')

# same the model
torch.save(model.state_dict(), "example.pt")
print("Model checkpoint saved to example.pt")