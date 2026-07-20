import torch
import torch.nn as nn
import torch.nn.functional as F

class WDCNN(nn.Module):
    """
    WDCNN (Wide Deep CNN) for vibration fault diagnosis.
    Based on Zhang et al. (2017): "Deep Convolutional Neural Networks with 
    Wide First-layer Kernels for Fault Diagnosis".
    """
    def __init__(self, in_channels=1, out_channels=10):
        super(WDCNN, self).__init__()
        
        # Layer 1: Wide kernel (64x1)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, out_channels)

    def extract_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.extract_features(x)
        x = self.fc2(x)
        return x

class TICNN(nn.Module):
    """
    TICNN (CNN with Training Interference) for vibration fault diagnosis.
    Based on Zhang et al. (2018): "A New Deep Learning Model for Fault 
    Diagnosis with Good Anti-Noise and Generalization Ability".
    """
    def __init__(self, in_channels=1, out_channels=10):
        super(TICNN, self).__init__()
        
        # Similar to WDCNN but with more dropout and potentially different stride
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 100)
        self.fc2 = nn.Linear(100, out_channels)

    def extract_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.extract_features(x)
        x = self.fc2(x)
        return x
