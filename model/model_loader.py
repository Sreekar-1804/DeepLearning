
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


class MultiTaskResNet34(nn.Module):
    def __init__(self, num_gender_classes=2, num_age_classes=4, pretrained=False):
        super(MultiTaskResNet34, self).__init__()

        weights = models.ResNet34_Weights.DEFAULT if pretrained else None

        self.backbone = models.resnet34(weights=weights)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_gender_classes)
        )

        self.age_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_age_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        gender_logits = self.gender_head(features)
        age_logits = self.age_head(features)

        return gender_logits, age_logits


def load_model(model_path, device):
    """
    Loads the trained multi-task ResNet34 model.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    num_gender_classes = checkpoint.get("num_gender_classes", 2)
    num_age_classes = checkpoint.get("num_age_classes", 4)

    model = MultiTaskResNet34(
        num_gender_classes=num_gender_classes,
        num_age_classes=num_age_classes,
        pretrained=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    gender_names = checkpoint.get("gender_names", {
        0: "Female",
        1: "Male"
    })

    age_group_names = checkpoint.get("age_group_names", {
        0: "Child",
        1: "Young Adult",
        2: "Adult",
        3: "Senior"
    })

    return model, gender_names, age_group_names
