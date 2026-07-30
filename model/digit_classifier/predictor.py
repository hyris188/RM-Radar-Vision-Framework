"""Inference wrapper for user-trained armor digit checkpoints."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from model.digit_classifier.model import build_model
from model.digit_classifier.transform import NormalTransform


class DigitClassifier:
    def __init__(
        self,
        model_type: str,
        weights_path: str,
        class_names=("1", "2", "3", "4", "S", "Q"),
    ):
        self.class_names = list(class_names)
        self.model = build_model(model_type, len(self.class_names), weights_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.model.eval()
        self.transform = NormalTransform(input_size=64)

    def predict(self, image, return_names=False):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            output = self.model(tensor)
        index = int(output.argmax(dim=1).item())
        label = self.class_names[index] if return_names else index
        return label, F.softmax(output, dim=1).squeeze().tolist()

    def predict_batch(self, images, return_names=False):
        tensor = torch.stack([self.transform(image) for image in images]).to(self.device)
        if self.device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            output = self.model(tensor)
        indices = output.argmax(dim=1).tolist()
        labels = [self.class_names[index] for index in indices] if return_names else indices
        return labels, F.softmax(output, dim=1).tolist()

    def get_class_names(self):
        return self.class_names
