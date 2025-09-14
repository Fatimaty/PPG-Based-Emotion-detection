
# model.py
# 1D CNN for PPG-based emotion recognition (5 classes).

import torch.nn as nn

class PPGEmotionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(PPGEmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(16)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
