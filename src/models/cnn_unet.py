import torch
import torch.nn as nn


class PneumoniaCNNWithUNet(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # ====================================================
        # TWO INPUT CHANNELS
        #
        # Channel 0 = original X-ray
        # Channel 1 = U-Net segmented X-ray
        # ====================================================

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Conv 1
            # ------------------------------------------------
            nn.Conv2d(
                2, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 2
            nn.Conv2d(
                16, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Conv 3
            # ------------------------------------------------
            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 4
            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Conv 5
            # ------------------------------------------------
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 6
            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # ------------------------------------------------
            # Conv 7
            # ------------------------------------------------
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 8
            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 9
            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            # Conv 10
            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification layer
        self.fc = nn.Linear(
            128,
            num_classes
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        # CNN feature maps
        features = self.features(x)

        # Global Average Pooling
        pooled = self.gap(
            features
        ).flatten(1)

        # Classification
        logits = self.fc(
            pooled
        )

        if return_features:
            return logits, features

        return logits