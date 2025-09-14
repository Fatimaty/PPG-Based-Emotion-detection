
# train.py
# Training pipeline for PPG-based emotion recognition.
# NOTE: Dataset is not public, add your dataset loading code here.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import PPGEmotionCNN

def train(data, labels, epochs=10, lr=0.001, batch_size=32, device="cpu"):
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PPGEmotionCNN(num_classes=len(torch.unique(labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for signals, y in loader:
            signals, y = signals.to(device), y.to(device)
            out = model(signals)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ppg_emotion_model.pth")
    print("Model saved as ppg_emotion_model.pth")

if __name__ == "__main__":
    print("Replace this with real dataset loading code.")
