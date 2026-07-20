import torch
import torch.nn as nn
import torch.nn.functional as F

class DCAE1D(nn.Module):
    """
    Denoising Convolutional Autoencoder 1D (DCAE1D) for vibration fault diagnosis.
    Focuses on non-supervised feature extraction.
    """
    def __init__(self, in_channels=1):
        super(DCAE1D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=16, stride=2, padding=7), # Output: L/2
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: L/4
            
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=3), # Output: L/8
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: L/16
            
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), # Output: L/32
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # Output: L/64
        )
        
        # Bottleneck (can be further refined if needed)
        # Assuming input length 2048, L/64 = 32. 64 * 32 = 2048 features.
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(16, in_channels, kernel_size=16, stride=8, padding=4, output_padding=0),
        )

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        encoded = self.encoder(x)
        return encoded.view(encoded.size(0), -1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # Add noise if training as denoising autoencoder (usually handled in training loop)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Resize to match input if needed, though ConvTranspose should be tuned
        if decoded.shape[-1] != x.shape[-1]:
             decoded = F.interpolate(decoded, size=x.shape[-1], mode='linear', align_corners=False)
        return decoded
