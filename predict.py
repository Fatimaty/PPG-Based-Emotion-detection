
# predict.py
# Run inference on PPG signal using saved model weights.

import torch
import numpy as np
from model import PPGEmotionCNN
from preprocess import bandpass_filter, normalize

def predict(signal, weights="ppg_emotion_model.pth", classes=None, fs=64, device="cpu"):
    model = PPGEmotionCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    # Preprocess
    signal = bandpass_filter(signal, fs=fs)
    signal = normalize(signal)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(signal_tensor)
        _, pred = torch.max(out, 1)
    return classes[pred]

if __name__ == "__main__":
    # Example usage
    emotions = ["happy", "sad", "angry", "calm", "fear"]
    dummy_signal = np.random.randn(256)
    label = predict(dummy_signal, classes=emotions)
    print("Predicted emotion:", label)
